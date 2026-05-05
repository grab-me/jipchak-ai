import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import datetime
import random

# ?꾨줈?앺듃 猷⑦듃 寃쎈줈 異붽? (examples ?대뜑???곸쐞 ?대뜑)
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, PROJECT_ROOT)

from three_jaw_grasp import GraspPipeline
from three_jaw_grasp.adapters import YoloSegGraspAdapter
from three_jaw_grasp.evaluator_chick import ChickEvaluator
from three_jaw_grasp.visualizer import draw_three_jaw_grasp

class YoloV8SegWrapper:
    """Ultralytics YOLOv8-seg 紐⑤뜽???뚯씠?꾨씪??洹쒓꺽??留욊쾶 ?섑븨"""
    def __init__(self, model_path):
        from ultralytics import YOLO
        # 紐⑤뜽 濡쒕뱶
        self.model = YOLO(model_path)
    
    def predict(self, rgb, depth=None, **kwargs):
        # conf=0.3 ?뺣룄濡??ㅼ젙?섏뿬 ?뺤떎??寃껊쭔 ?먯?
        results = self.model.predict(rgb, conf=0.3, verbose=False)
        output = []
        for r in results:
            if r.masks is not None:
                # 紐⑤뜽 異쒕젰??留덉뒪???곗씠?곕? ?먮낯 ?대?吏 ?ш린濡??낆깦?뚮쭅
                # (YOLOv8-seg??湲곕낯?곸쑝濡???? ?댁긽?꾩쓽 留덉뒪?щ? 異쒕젰?섎?濡?r.masks.data媛 ?먮낯 ?ш린?몄? ?뺤씤 ?꾩슂)
                masks = r.masks.data.cpu().numpy()
                boxes = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                
                # ?대?吏 ?ш린??留욎떠 留덉뒪??由ъ궗?댁쫰 (?꾩슂??寃쎌슦)
                h, w = rgb.shape[:2]
                
                for i in range(len(masks)):
                    m = masks[i]
                    # 留뚯빟 留덉뒪???ш린媛 ?먮낯怨??ㅻⅤ硫?由ъ궗?댁쫰
                    if m.shape[0] != h or m.shape[1] != w:
                        import cv2
                        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                    
                    output.append({
                        'mask': m,
                        'box': boxes[i].tolist(),
                        'score': float(confs[i])
                    })
        return output

import datetime

def run_test():
    # 1. 寃쎈줈 ?ㅼ젙
    # dataset/model_chick2_v2/best2.pt 寃쎈줈
    MODEL_PATH = os.path.join(PROJECT_ROOT, 'dataset', 'model_chick2_v2', 'best2.pt')
    DATASET_DIR = os.path.join(PROJECT_ROOT, 'dataset', 'model_chick2_v2', 'test_images_sample')
    
    print(f"--- Chick V2 AI ?뚯뒪???쒖옉 ---")
    print(f"紐⑤뜽 寃쎈줈: {MODEL_PATH}")
    print(f"?곗씠??寃쎈줈: {DATASET_DIR}")

    if not os.path.exists(MODEL_PATH):
        print(f"[?ㅻ쪟] 紐⑤뜽 ?뚯씪??李얠쓣 ???놁뒿?덈떎: {MODEL_PATH}")
        return
    if not os.path.exists(DATASET_DIR):
        print(f"[?ㅻ쪟] ?곗씠?곗뀑 ?대뜑瑜?李얠쓣 ???놁뒿?덈떎: {DATASET_DIR}")
        return

    # 2. ?뚯씠?꾨씪??援ъ꽦
    print("\nAI ?뚯씠?꾨씪??珥덇린??以?(YOLOv8-seg + ChickStrategy)...")
    try:
        model = YoloV8SegWrapper(MODEL_PATH)
        adapter = YoloSegGraspAdapter()
        evaluator = ChickEvaluator(jaw_count=3) # ?곕━媛 留뚮뱺 理쒖떊 ?꾨왂 ?됯?湲??곸슜
        
        # detector=None?쇰줈 ?ㅼ젙?섏뿬 ?꾩껜 ?대?吏?먯꽌 吏곸젒 ?멸렇硫섑뀒?댁뀡 ?섑뻾
        pipeline = GraspPipeline(model=model, adapter=adapter, evaluator=evaluator)
    except Exception as e:
        print(f"[?ㅻ쪟] 珥덇린???ㅽ뙣: {e}")
        return

    # 3. ?뚯뒪???대?吏 寃??    image_paths = glob.glob(os.path.join(DATASET_DIR, "*.png"))
    if not image_paths:
        print("[寃쎄퀬] ?뚯뒪?명븷 ?대?吏媛 ?놁뒿?덈떎.")
        return
        
    # ?쒕뜡?섍쾶 5???섑뵆留?    test_count = min(5, len(image_paths))
    image_paths = random.sample(image_paths, test_count)
    print(f"珥?{len(image_paths)}??以??쒕뜡?섍쾶 {test_count}?μ쓣 異붿텧?섏뿬 ?뚯뒪?명빀?덈떎.")

    # 4. ????대뜑 ?앹꽦 (?좎쭨_踰덊샇 ?뺤떇?쇰줈 ?먮룞 利앷?)
    output_base = os.path.join(THIS_DIR, "output")
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    
    # ?대떦 ?좎쭨濡??쒖옉?섎뒗 ?대뜑 紐⑸줉 ?뺤씤
    existing_dirs = glob.glob(os.path.join(output_base, f"chick_v2_{date_str}_*"))
    max_num = 0
    for d in existing_dirs:
        try:
            num_str = os.path.basename(d).split('_')[-1]
            max_num = max(max_num, int(num_str))
        except (ValueError, IndexError):
            continue
    
    new_num = max_num + 1
    save_dir = os.path.join(output_base, f"chick_v2_{date_str}_{new_num}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"[?뚮┝] 寃곌낵媛 ?ㅼ쓬 ?대뜑????λ맗?덈떎: {save_dir}")

    # 5. 猷⑦봽 ?ㅽ뻾
    for i, img_path in enumerate(image_paths[:test_count]):
        fname = os.path.basename(img_path)
        print(f"\n[{i+1}/{test_count}] {fname} 泥섎━ 以?..")
        
        try:
            # ?대?吏 濡쒕뱶
            img_pil = Image.open(img_path).convert('RGB')
            rgb = np.array(img_pil)
            
            # Depth 媛???앹꽦 (30cm 嫄곕━)
            depth = np.full((rgb.shape[0], rgb.shape[1]), 0.30, dtype=np.float32)
            
            # ?뚯씠?꾨씪???ㅽ뻾
            best = pipeline.predict_best(rgb, depth)
            
            # ?곸꽭 ?먯닔 ?띾뱷
            detail = evaluator.score_detail(best)
            
            print(f" > ?먯? ?깃났! 理쒖쥌 ?먯닔: {detail['total']:.3f}")
            print(f" > 二쇱슂 吏??- TopBias: {detail.get('top_bias', {}).get('raw', 0):.2f}, Hooking: {detail.get('hooking', {}).get('raw', 0):.2f}")
            
            # 寃곌낵 ?쒓컖??            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            draw_three_jaw_grasp(ax, rgb, best, detail, model_name="Chick_V2_AI")
            plt.title(f"Chick V2 AI Test: {fname} (Score: {detail['total']:.3f})")
            
            # 寃곌낵 ???            save_path = os.path.join(save_dir, f"result_{fname}")
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f" > 寃곌낵 ????꾨즺: {save_path}")
            
            plt.show()
            plt.close()
            
        except Exception as e:
            print(f" > [{fname}] 泥섎━ 以??ㅻ쪟 諛쒖깮: {e}")

    print("\n--- 紐⑤뱺 ?뚯뒪?멸? ?꾨즺?섏뿀?듬땲?? ---")

if __name__ == "__main__":
    run_test()

