import os
from llim_orb_ransac_pipeline import evaluate_sequence
from utils.evaluation import load_tum_rgb_list,load_darkzurich_rgb_list
from utils.enhancer import load_model,  enhance_image_cv
import cv2
# =====================================================
# 示例增强函数（替换成你的）
# =====================================================
def enhance(img):

    return img

def resize_image_cv(image_bgr, target_size=(600, 400)):
    return cv2.resize(image_bgr, target_size, interpolation=cv2.INTER_CUBIC)



# =====================================================
# 主测试函数
# =====================================================
def main():

    # sequence_path = "datasets/lle/darkZurich/set1"  # 替换成你的 TUM 数据集路径
    sequence_path  = "datasets\\RealData\\video2_walk"
    # image_list = load_tum_rgb_list(sequence_path)
    image_list = load_darkzurich_rgb_list(sequence_path)

    print("Total frames:", len(image_list))

    # 可选：限制帧数加快测试
    # image_list = image_list[:10]

    # ===============================
    # 1Raw
    # ===============================
    print("\nRunning RAW evaluation...")
    results_raw = evaluate_sequence(image_list,
                                    enhance=False)

    # ===============================
    # Always Enhanced
    # ===============================
    print("\nRunning ENHANCED evaluation...")
    results_enhanced = evaluate_sequence(
        image_list,
        enhance=True,
        enhance_mode='every',
        save_vis=True, vis_save_dir = "experiments\\results\\RealData\\video2_walk_enhanced"
    )
    
    # ===============================
    # Adaptive Enhanced
    # ===============================
    print("\nRunning ADAPTIVE ENHANCED evaluation...")
    results_adaptive = evaluate_sequence(
        image_list,
        enhance=True,
        enhance_mode='adaptive',
        tau_mu=50,
        tau_I=10
    )

    # ===============================
    # 打印对比结果
    # ===============================
    print("\n============================")
    print("Proxy Evaluation on video2_walk")
    print("============================")

    print("\nRAW:")
    for k, v in results_raw.items():
        print(f"{k}: {v:.4f}")

    print("\nENHANCED:")
    for k, v in results_enhanced.items():
        print(f"{k}: {v:.4f}")

    print("\nADAPTIVE ENHANCED:")
    for k, v in results_adaptive.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()