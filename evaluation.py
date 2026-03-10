from proxy_slam_evaluation import evaluate_sequence

# image_list 是按顺序的图片路径列表

# Raw
results_raw = evaluate_sequence(image_list)

# Always enhance
results_enhanced = evaluate_sequence(image_list, enhance_fn=enhance)

print("Raw:", results_raw)
print("Enhanced:", results_enhanced)