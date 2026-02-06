import av
import glob
from tqdm import tqdm

roots = [
    "/home/dx/waylen/SARM/transformed_datasets/pick_cube_sim_sparse",
    "/home/dx/waylen/SARM/transformed_datasets/pick_cube_sim_dense"
]

print("正在进行深度视频检查（扫描所有帧）...")
for root in roots:
    videos = glob.glob(f"{root}/**/*.mp4", recursive=True)
    for v_path in tqdm(videos, desc=f"Scanning {root.split('/')[-1]}"):
        try:
            # container = av.open(v_path)
            # stream = container.streams.video[0]
            # # 模拟遍历所有包，不解码图像（速度快且能发现截断）
            # for packet in container.demux(stream):
            #     pass
            # container.close()
            with av.open(v_path) as c:
                for frame in c.decode(video=0):
                    break
        except Exception as e:
            print(f"\n[发现坏文件] {v_path}")
            print(f"错误: {e}")
            # 建议直接重新生成数据，不要试图修复单个文件
