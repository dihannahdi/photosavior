"""Quick test of CLIP adversarial attack."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.clip_adversarial import CLIPAdversarialShield

# Create a test image  
img = np.random.RandomState(42).rand(256, 256, 3)
shield = CLIPAdversarialShield(strength='moderate')
protected, report = shield.protect(img, verbose=True)

print()
print("=== RESULTS ===")
cos = report['cosine_similarity']
print(f"Cosine similarity: {cos:.4f}")
print(f"Feature distance:  {report['feature_distance']:.2f}")
print(f"PSNR:              {report['psnr_db']:.1f} dB")
linf = report['linf']
print(f"L-inf:             {linf:.4f} ({linf*255:.1f}/255)")
print(f"CLIP embedding shifted by {(1-cos)*100:.1f}%")

if cos < 0.5:
    print("\nSUCCESS: CLIP sees a COMPLETELY DIFFERENT image!")
elif cos < 0.8:
    print("\nSUCCESS: CLIP sees a SIGNIFICANTLY different image!")
elif cos < 0.95:
    print("\nPARTIAL: CLIP sees a somewhat different image.")
else:
    print("\nFAILED: CLIP still sees the same image.")
