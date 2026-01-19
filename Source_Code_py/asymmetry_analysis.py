import numpy as np
import os

PATH_TO_DATA = r"."
DC_OFFSET = 128

def compute_skewness(signal):
    n = len(signal)
    if n < 3: return 0.0
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    if std_val == 0: return 0.0
    
    skew = np.sum((signal - mean_val)**3) / ((n - 1) * (std_val**3))
    return skew

def calculate_asymmetry_percentage(val1, val2):
    numerator = abs(val1 - val2)
    denominator = max(abs(val1), abs(val2))
    
    if denominator == 0: return 0.0
    return (numerator / denominator) * 100.0

def get_classification(kas_value):
    if kas_value < 10: return "Class 1 (Healthy)"
    if kas_value < 20: return "Class 2 (Mild)"
    if kas_value < 40: return "Class 3 (Moderate)"
    return "Class 4 (High)"

def process_file(file_path):
    try:
        data = np.load(file_path)
        
        rms_channels = []
        skew_channels = []
        
        for i in range(8):
            sig = data[i].astype(int) - DC_OFFSET
            
            rms = np.sqrt(np.mean(sig**2))
            rms_channels.append(rms)
            
            skew = compute_skewness(sig)
            skew_channels.append(skew)
            
        flex_idx = np.argmax(rms_channels)
        ext_idx = (flex_idx + 4) % 8
        
        rms_flex = rms_channels[flex_idx]
        rms_ext = rms_channels[ext_idx]
        
        skew_flex = skew_channels[flex_idx]
        skew_ext = skew_channels[ext_idx]
        
        kas_rms = calculate_asymmetry_percentage(rms_flex, rms_ext)
        kas_skew = calculate_asymmetry_percentage(skew_flex, skew_ext)
        
        diag_class = get_classification(kas_rms)
        
        return kas_rms, kas_skew, diag_class, flex_idx, ext_idx
        
    except Exception:
        return None

def extract_subject_id(filename):
    try:
        parts = filename.split('_')
        return int(parts[1])
    except:
        return 999999

def run_analysis():
    if not os.path.exists(PATH_TO_DATA):
        print("Error: The specified path does not exist!")
        return

    files = [f for f in os.listdir(PATH_TO_DATA) if f.endswith('.npy')]
    files.sort(key=extract_subject_id)
    
    stats_rms = {0: [], 1: [], 2: []}
    stats_skew = {0: [], 1: [], 2: []}
    
    print("\n" + "="*115)
    print(f"PART 1: DETAILED REPORT (Sample of first 15 files)")
    print("="*115)
    print(f"{'FILENAME':<22} | {'MOV':<3} | {'K_AS ASYM(%)':<12} | {' K_AS ASYM CLASS':<18} || {'SKEWNESS ASYM(%)':<12} | {'SKEWNESS ASYM CLASS':<18}")
    print("-" * 115)

    count_printed = 0
    limit_print = 15

    for fname in files:
        parts = fname.split('_')
        
        if len(parts) > 2 and parts[2].isdigit():
            cls = int(parts[2])
            full_path = os.path.join(PATH_TO_DATA, fname)
            
            result = process_file(full_path)
            
            if result is not None:
                kas_rms, kas_skew, diag_rms, flex_ch, ext_ch = result
                diag_skew = get_classification(kas_skew)

                if cls in stats_rms:
                    stats_rms[cls].append(kas_rms)
                    stats_skew[cls].append(kas_skew)
                
                if count_printed < limit_print:
                    print(f"{fname:<22} | {cls:<3} | {kas_rms:<12.2f} | {diag_rms:<18} || {kas_skew:<12.2f} | {diag_skew:<18}")
                    count_printed += 1


    print("\n" + "="*100)
    print("PART 2: FINAL AGGREGATED STATISTICS (AVERAGES)")
    print("="*100)
    print(f"{'MOVEMENT':<10} | {'SAMPLES':<8} | {'AVG K_AS ASYM(%)':<15} | {'AVG K_AS ASYM CLASS':<18} || {'AVG SKEWNESS ASYM(%)':<15} | {'AVG SKEWNESS CLASS':<15}")
    print("-" * 100)
    
    for cls_id in [0, 1, 2]:
        vals_rms = stats_rms[cls_id]
        vals_skew = stats_skew[cls_id]
        
        if len(vals_rms) > 0:
            avg_rms = np.mean(vals_rms)
            avg_skew = np.mean(vals_skew)
            avg_class_str = get_classification(avg_rms)
            avg_class_skew = get_classification(avg_skew)
            
            print(f"Type {cls_id:<5} | {len(vals_rms):<8} | {avg_rms:<15.2f} | {avg_class_str:<18} || {avg_skew:<15.2f} | {avg_class_skew:<18}")
        else:
            print(f"Type {cls_id:<5} | 0        | N/A")
            
    print("-" * 100)

if __name__ == "__main__":
    run_analysis()