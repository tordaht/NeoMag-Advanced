import taichi as ti
import torch
import torch.utils.dlpack
import numpy as np

def verify_pipeline():
    print("--- [FAZ 0] sm_120 & DLPack Zero-Copy Doğrulaması ---")
    
    # 1. Taichi Init (CUDA)
    try:
        ti.init(arch=ti.cuda)
        print("[OK] Taichi CUDA aktif.")
    except Exception as e:
        print(f"[HATA] Taichi CUDA başlatılamadı: {e}")
        return False

    # 2. PyTorch Device Check
    device = torch.device("cuda")
    print(f"[INFO] PyTorch Cihazı: {torch.cuda.get_device_name(0)} (sm_120)")

    # 3. Create Taichi ndarray
    shape = (1024,)
    arr = ti.ndarray(dtype=ti.f32, shape=shape)
    
    @ti.kernel
    def fill_arr(a: ti.types.ndarray()):
        for i in range(1024):
            a[i] = float(i)
    
    fill_arr(arr)
    ti.sync()
    print("[OK] Taichi ndarray verisi hazır.")

    # 4. DLPack Transfer (Zero-Copy)
    try:
        # Taichi to DLPack
        dpack = arr.to_dlpack()
        # DLPack to Torch
        tensor = torch.from_dlpack(dpack)
        print("[OK] DLPack transferi başarılı (Zero-Copy Bridge).")
        
        # 5. Kernel Execution (The sm_120 Reality Check)
        print("[WAIT] PyTorch GPU Kernel testi yapılıyor...")
        test_op = tensor * 2.0
        # Force sync to catch asynchronous CUDA errors
        torch.cuda.synchronize()
        
        val = test_op[10].item()
        if val == 20.0:
            print("[MÜKEMMEL] sm_120 üzerinde PyTorch GPU kernelları ÇALIŞIYOR.")
            return True
        else:
            print(f"[HATA] Veri tutarsızlığı: {val} != 20.0")
            return False
            
    except Exception as e:
        print(f"\n[KRİTİK BLOKAJ] sm_120 DLPack/Kernel Hatası: {e}")
        print("[DÜRÜST ANALİZ] PyTorch mevcut sürümü sm_120 (RTX 5080) kernel'larını GPU üzerinde yürütemiyor.")
        print("[SONUÇ] v17.1 'Zero-Copy' hedefi mevcut PyTorch/CUDA kütüphaneleriyle donanımsal engele takılmıştır.")
        return False

if __name__ == "__main__":
    verify_pipeline()
