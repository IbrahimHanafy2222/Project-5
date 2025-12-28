from flask import Flask, render_template, request, jsonify
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# ============ BIT GENERATION ============
def generate_bits(N):
    return np.random.randint(0, 2, N)

# ============ BPSK MODULATION / DEMODULATION ============
def bpsk_mod(bits):
    return 2*bits - 1 # 0 -> -1 and 1 -> 1

def bpsk_demod(received): # (>0 -> 1) (<0 -> 0)
    return (received >= 0).astype(int)

# ============ CHANNELS ============
def awgn_channel(signal, EbN0_dB):
    EbN0 = 10**(EbN0_dB / 10) # converts Eb/N0 from dB to linear scale
    sigma = np.sqrt(1 / (2 * EbN0)) # noise standard deviation
    noise = sigma * np.random.randn(len(signal)) 
    return signal + noise # adding noise to the signal

def rayleigh_channel(signal, EbN0_dB):
    EbN0 = 10**(EbN0_dB / 10)
    sigma = np.sqrt(1 / (2 * EbN0))
    h = np.sqrt(                                           # fading coefficient
        (np.random.randn(len(signal))**2 +
         np.random.randn(len(signal))**2) / 2
    )
    noise = sigma * np.random.randn(len(signal))
    received = h * signal + noise
    return received / h

# ============ REPETITION CODE ============
def repetition_encode(bits, n): # repetition factor n
    return np.repeat(bits, n)

def repetition_decode(bits, n):
    bits = bits.reshape(-1, n)
    return (np.sum(bits, axis=1) > n/2).astype(int) # majority

# ============ HAMMING (7,4) CODE ============
def hamming74_encode(bits):
    G = np.array([
        [1, 0, 0, 0, 1, 1, 0],
        [0, 1, 0, 0, 1, 0, 1],
        [0, 0, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1]
    ])
    bits = bits.reshape(-1, 4)
    coded = (bits @ G) % 2 # flips erroneous bit
    return coded.flatten()

def hamming74_decode(bits):
    H = np.array([
        [1, 1, 0, 1, 1, 0, 0],
        [1, 0, 1, 1, 0, 1, 0],
        [0, 1, 1, 1, 0, 0, 1]
    ])
    bits = bits.reshape(-1, 7)
    decoded_bits = []
    
    for word in bits:
        syndrome = (H @ word) % 2
        if np.any(syndrome):
            for i in range(7):
                if np.array_equal(H[:, i], syndrome):
                    word[i] ^= 1
                    break
        decoded_bits.append(word[:4]) 
    
    return np.array(decoded_bits).flatten()

# ============ HAMMING (15,11) CODE ============
G1511 = np.array([
    [1,0,0,0,0,0,0,0,0,0,0,1,1,0,1],
    [0,1,0,0,0,0,0,0,0,0,0,1,0,1,1],
    [0,0,1,0,0,0,0,0,0,0,0,0,1,1,1],
    [0,0,0,1,0,0,0,0,0,0,0,1,1,1,0],
    [0,0,0,0,1,0,0,0,0,0,0,1,0,0,1],
    [0,0,0,0,0,1,0,0,0,0,0,0,1,0,1],
    [0,0,0,0,0,0,1,0,0,0,0,0,0,1,1],
    [0,0,0,0,0,0,0,1,0,0,0,1,1,0,0],
    [0,0,0,0,0,0,0,0,1,0,0,1,0,1,0],
    [0,0,0,0,0,0,0,0,0,1,0,0,1,1,0],
    [0,0,0,0,0,0,0,0,0,0,1,1,0,0,0]
])

# Parity Check Matrix H (4x15)
H1511 = np.array([
    [1,1,0,1,1,0,0,1,1,0,0,1,0,0,0],
    [1,0,1,1,0,1,0,1,0,1,0,0,1,0,0],
    [0,1,1,1,0,0,1,0,1,1,0,0,0,1,0],
    [1,1,1,0,1,1,1,0,0,0,1,0,0,0,1]
])

def hamming1511_encode(bits):
    extra = len(bits) % 11
    if extra != 0:
        bits = bits[:-extra]
        
    bits_reshaped = bits.reshape(-1, 11)
    coded = (bits_reshaped @ G1511) % 2
    return coded.flatten()

def hamming1511_decode(bits):
    bits_reshaped = bits.reshape(-1, 15)
    decoded = []
    
    # Calculate syndromes: (N_words x 4)
    syndromes = (bits_reshaped @ H1511.T) % 2
    
    for i in range(len(bits_reshaped)):
        syn = syndromes[i]
        if np.any(syn):

            for col_idx in range(15):
                if np.array_equal(H1511[:, col_idx], syn):
                    bits_reshaped[i, col_idx] ^= 1 # Correct error
                    break
                    
        decoded.append(bits_reshaped[i, :11]) # Extract systematic bits
    
    return np.array(decoded).flatten()


# ============ BER CALCULATION ============
def ber(bits, bits_hat): # error bits/total bits
    return np.mean(bits != bits_hat)

def adjust_EbN0(EbN0_dB, rate): # fairy comparison
    return EbN0_dB + 10*np.log10(rate)

# ============ SIMULATION FUNCTION ============
def simulate(channel_type, use_coding, code_type, code_param, snr_range, N=100000):
    ber_values = []
    
    for EbN0_dB in snr_range:
        bits = generate_bits(N)
        
        if channel_type == 'awgn':
            tx = bpsk_mod(bits)
            rx = awgn_channel(tx, EbN0_dB)
            ber_values.append(ber(bits, bpsk_demod(rx)))
            
        elif channel_type == 'rayleigh':
            if not use_coding:
                tx = bpsk_mod(bits)
                rx = rayleigh_channel(tx, EbN0_dB)
                ber_values.append(ber(bits, bpsk_demod(rx)))
            else:
                if code_type == 'repetition':
                    n = code_param
                    rate = 1/n
                    coded = repetition_encode(bits, n)
                    tx = bpsk_mod(coded)
                    rx = rayleigh_channel(tx, adjust_EbN0(EbN0_dB, rate))
                    decoded = repetition_decode(bpsk_demod(rx), n)
                    ber_values.append(ber(bits, decoded))
                    
                elif code_type == 'hamming':
                    m = code_param
                    if m == 3:  # (7,4)
                        k, n, rate = 4, 7, 4/7
                        bits_h = bits[:len(bits) - len(bits)%k]
                        coded = hamming74_encode(bits_h)
                        tx = bpsk_mod(coded)
                        rx = rayleigh_channel(tx, adjust_EbN0(EbN0_dB, rate))
                        decoded = hamming74_decode(bpsk_demod(rx))
                        ber_values.append(ber(bits_h, decoded))
                    elif m == 4:  # (15,11)
                        k, n, rate = 11, 15, 11/15
                        bits_h = bits[:len(bits) - len(bits)%k]
                        coded = hamming1511_encode(bits_h)
                        tx = bpsk_mod(coded)
                        rx = rayleigh_channel(tx, adjust_EbN0(EbN0_dB, rate))
                        decoded = hamming1511_decode(bpsk_demod(rx))
                        ber_values.append(ber(bits_h, decoded))
    
    return ber_values

# ============ FLASK ROUTES ============
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/simulate', methods=['POST'])
def run_simulation():
    data = request.json
    
    channel_type = data['channel_type']
    use_coding = data.get('use_coding', False)
    code_type = data.get('code_type', 'repetition')
    code_param = data.get('code_param', 3)
    snr_min = data['snr_min']
    snr_max = data['snr_max']
    snr_step = data['snr_step']
    
    snr_range = np.arange(snr_min, snr_max + snr_step, snr_step)
    
    ber_values = simulate(channel_type, use_coding, code_type, code_param, snr_range)
    
    # Generate plot with compact size
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    
    ax.semilogy(snr_range, ber_values, 'o-', linewidth=2, markersize=6)
    ax.grid(True, which='both', alpha=0.3, linestyle='--')
    ax.set_xlabel('Eb/N0 (dB)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Bit Error Rate (BER)', fontsize=11, fontweight='bold')
    
    # Create title based on configuration
    if channel_type == 'awgn':
        title = 'AWGN Channel'
    else:
        if use_coding:
            if code_type == 'repetition':
                title = f'Rayleigh with Repetition (1/{code_param})'
            else:
                n = 2**code_param - 1
                k = n - code_param
                title = f'Rayleigh with Hamming ({n},{k})'
        else:
            title = 'Rayleigh Channel (No Coding)'
    
    ax.set_title(f'BER Performance - {title}', fontsize=12, fontweight='bold', pad=12)
    
    # Set proper y-axis limits to show full range
    min_ber = min(ber_values)
    max_ber = max(ber_values)
    ax.set_ylim([max(min_ber * 0.1, 1e-6), min(max_ber * 10, 1)])
    
    # Set x-axis limits with padding
    ax.set_xlim([snr_range[0] - 0.5, snr_range[-1] + 0.5])
    
    # Add margins
    fig.subplots_adjust(left=0.13, right=0.95, top=0.91, bottom=0.12)
    
    # Convert plot to base64 string
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=90, bbox_inches='tight', pad_inches=0.15)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return jsonify({
        'snr_range': snr_range.tolist(),
        'ber_values': ber_values,
        'plot_url': plot_url,
        'title': title
    })

@app.route('/generate_all_figures', methods=['POST'])
def generate_all_figures():
    """Generate all three required figures"""
    data = request.json
    snr_min = data.get('snr_min', 0)
    snr_max = data.get('snr_max', 15)
    snr_step = data.get('snr_step', 1)
    
    snr_range = np.arange(snr_min, snr_max + snr_step, snr_step)
    
    print("Generating all figures...")
    
    # Simulate all systems
    print("Simulating AWGN (No Coding)...")
    ber_awgn = simulate('awgn', False, None, None, snr_range)
    
    print("Simulating Rayleigh (No Coding)...")
    ber_rayleigh = simulate('rayleigh', False, None, None, snr_range)
    
    print("Simulating Repetition (1/3)...")
    ber_rep3 = simulate('rayleigh', True, 'repetition', 3, snr_range)
    
    print("Simulating Repetition (1/5)...")
    ber_rep5 = simulate('rayleigh', True, 'repetition', 5, snr_range)
    
    print("Simulating Hamming (7,4)...")
    ber_ham74 = simulate('rayleigh', True, 'hamming', 3, snr_range)
    
    print("Simulating Hamming (15,11)...")
    ber_ham1511 = simulate('rayleigh', True, 'hamming', 4, snr_range)
    
    # ========== FIGURE 1: AWGN vs Rayleigh (No Coding) ==========
    fig1 = plt.figure(figsize=(8, 5.5))
    ax1 = fig1.add_subplot(111)
    
    ax1.semilogy(snr_range, ber_awgn, 'b-o', linewidth=2, markersize=6, label='AWGN Channel', markeredgewidth=1.2)
    ax1.semilogy(snr_range, ber_rayleigh, 'r-s', linewidth=2, markersize=6, label='Rayleigh Fading Channel', markeredgewidth=1.2)
    ax1.grid(True, which='both', alpha=0.3, linestyle='--', linewidth=0.7)
    ax1.set_xlabel('Eb/N0 (dB)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Bit Error Rate (BER)', fontsize=11, fontweight='bold')
    ax1.set_title('Figure 1: BER Performance - AWGN vs Rayleigh Fading (No Coding)', 
              fontsize=12, fontweight='bold', pad=12)
    ax1.legend(fontsize=10, loc='best', framealpha=0.95, edgecolor='gray')
    
    # Dynamic y-axis limits
    all_ber = ber_awgn + ber_rayleigh
    min_ber = min(all_ber)
    max_ber = max(all_ber)
    ax1.set_ylim([max(min_ber * 0.1, 1e-6), min(max_ber * 10, 1)])
    ax1.set_xlim([snr_range[0] - 0.5, snr_range[-1] + 0.5])
    
    # Add margins
    fig1.subplots_adjust(left=0.13, right=0.95, top=0.91, bottom=0.12)
    
    img1 = io.BytesIO()
    plt.savefig(img1, format='png', dpi=100, bbox_inches='tight', pad_inches=0.15)
    img1.seek(0)
    plot1_url = base64.b64encode(img1.getvalue()).decode()
    plt.close()
    
    # ========== FIGURE 2: Channel Coding Cases (4 cases) ==========
    fig2 = plt.figure(figsize=(8, 5.5))
    ax2 = fig2.add_subplot(111)
    
    ax2.semilogy(snr_range, ber_rep3, 'g-o', linewidth=2, markersize=6, label='Repetition (1/3)', markeredgewidth=1.2)
    ax2.semilogy(snr_range, ber_rep5, 'm-^', linewidth=2, markersize=6, label='Repetition (1/5)', markeredgewidth=1.2)
    ax2.semilogy(snr_range, ber_ham74, 'c-s', linewidth=2, markersize=6, label='Hamming (7,4)', markeredgewidth=1.2)
    ax2.semilogy(snr_range, ber_ham1511, 'y-d', linewidth=2, markersize=6, label='Hamming (15,11)', markeredgewidth=1.2, markerfacecolor='gold')
    ax2.grid(True, which='both', alpha=0.3, linestyle='--', linewidth=0.7)
    ax2.set_xlabel('Eb/N0 (dB)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Bit Error Rate (BER)', fontsize=11, fontweight='bold')
    ax2.set_title('Figure 2: BER Performance - Rayleigh Fading with Channel Coding', 
              fontsize=12, fontweight='bold', pad=12)
    ax2.legend(fontsize=10, loc='best', framealpha=0.95, edgecolor='gray')
    
    # Dynamic y-axis limits
    all_ber = ber_rep3 + ber_rep5 + ber_ham74 + ber_ham1511
    min_ber = min(all_ber)
    max_ber = max(all_ber)
    ax2.set_ylim([max(min_ber * 0.1, 1e-6), min(max_ber * 10, 1)])
    ax2.set_xlim([snr_range[0] - 0.5, snr_range[-1] + 0.5])
    
    # Add margins
    fig2.subplots_adjust(left=0.13, right=0.95, top=0.91, bottom=0.12)
    
    img2 = io.BytesIO()
    plt.savefig(img2, format='png', dpi=100, bbox_inches='tight', pad_inches=0.15)
    img2.seek(0)
    plot2_url = base64.b64encode(img2.getvalue()).decode()
    plt.close()
    
    # ========== FIGURE 3: Complete Comparison ==========
    fig3 = plt.figure(figsize=(9, 6))
    ax3 = fig3.add_subplot(111)
    
    ax3.semilogy(snr_range, ber_rayleigh, 'r-s', linewidth=2.2, markersize=7, 
                 label='Rayleigh (No Coding)', markerfacecolor='white', markeredgewidth=1.5)
    ax3.semilogy(snr_range, ber_rep3, 'g-o', linewidth=2, markersize=6, label='Repetition (1/3)', markeredgewidth=1.2)
    ax3.semilogy(snr_range, ber_rep5, 'm-^', linewidth=2, markersize=6, label='Repetition (1/5)', markeredgewidth=1.2)
    ax3.semilogy(snr_range, ber_ham74, 'c-s', linewidth=2, markersize=6, label='Hamming (7,4)', markeredgewidth=1.2)
    ax3.semilogy(snr_range, ber_ham1511, 'y-d', linewidth=2, markersize=6, label='Hamming (15,11)', markeredgewidth=1.2, markerfacecolor='gold')
    ax3.semilogy(snr_range, ber_awgn, 'b-o', linewidth=2, markersize=6, 
                 label='AWGN (Reference)', alpha=0.7, linestyle='--', markeredgewidth=1.2)
    
    ax3.grid(True, which='both', alpha=0.3, linestyle='--', linewidth=0.7)
    ax3.set_xlabel('Eb/N0 (dB)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Bit Error Rate (BER)', fontsize=11, fontweight='bold')
    ax3.set_title('Figure 3: BER Comparison - Coded vs Uncoded Systems\n(Fair Comparison: Same Information Bit Energy)', 
              fontsize=12, fontweight='bold', pad=12)
    ax3.legend(fontsize=9, loc='best', ncol=2, framealpha=0.95, edgecolor='gray')
    
    # Dynamic y-axis limits
    all_ber = ber_awgn + ber_rayleigh + ber_rep3 + ber_rep5 + ber_ham74 + ber_ham1511
    min_ber = min(all_ber)
    max_ber = max(all_ber)
    ax3.set_ylim([max(min_ber * 0.1, 1e-6), min(max_ber * 10, 1)])
    ax3.set_xlim([snr_range[0] - 0.5, snr_range[-1] + 0.5])
    
    # Add text box with explanation
    textstr = 'Note: Eb/N0 adjusted by code rate\nfor fair comparison'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.75, edgecolor='gray', linewidth=1.2)
    ax3.text(0.025, 0.025, textstr, transform=ax3.transAxes, fontsize=9,
             verticalalignment='bottom', bbox=props, fontweight='bold')
    
    # Add margins
    fig3.subplots_adjust(left=0.12, right=0.95, top=0.90, bottom=0.11)
    
    img3 = io.BytesIO()
    plt.savefig(img3, format='png', dpi=100, bbox_inches='tight', pad_inches=0.15)
    img3.seek(0)
    plot3_url = base64.b64encode(img3.getvalue()).decode()
    plt.close()
    
    # Generate summary table
    idx_10db = np.where(snr_range == 10)[0]
    if len(idx_10db) > 0:
        idx = idx_10db[0]
        summary = {
            'AWGN (No Coding)': f"{ber_awgn[idx]:.6e}",
            'Rayleigh (No Coding)': f"{ber_rayleigh[idx]:.6e}",
            'Repetition (1/3)': f"{ber_rep3[idx]:.6e}",
            'Repetition (1/5)': f"{ber_rep5[idx]:.6e}",
            'Hamming (7,4)': f"{ber_ham74[idx]:.6e}",
            'Hamming (15,11)': f"{ber_ham1511[idx]:.6e}"
        }
    else:
        summary = {}
    
    print("All figures generated successfully!")
    
    return jsonify({
        'figure1': plot1_url,
        'figure2': plot2_url,
        'figure3': plot3_url,
        'summary': summary
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)