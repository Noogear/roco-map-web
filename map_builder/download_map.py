import os
import requests
from PIL import Image
from io import BytesIO
import time
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ================= 核心配置区域 =================
# 已经为你替换好了真实的 URL 规律
BASE_URL = "https://wiki-dev-patch-oss.oss-cn-hangzhou.aliyuncs.com/res/lkwg/map-3.0/7/tile-{x}_{y}.png"

X_MIN = -12
X_MAX = 11
Y_MIN = -11
Y_MAX = 11

TILE_SIZE = 256  # 瓦片图标准尺寸
OUT_DIR = os.path.join(_ROOT, "output")


# ================================================

def download_and_stitch():
    # 1. 计算最终画布的宽高
    width = (X_MAX - X_MIN + 1) * TILE_SIZE
    height = (Y_MAX - Y_MIN + 1) * TILE_SIZE

    print(f"[开始] 画布尺寸: {width} x {height} 像素，共 {(X_MAX-X_MIN+1)*(Y_MAX-Y_MIN+1)} 块瓦片")
    result_image = Image.new("RGBA", (width, height), (0, 0, 0, 0))

    # 2. 伪装请求头，防止被服务器识别为爬虫拦截
    session = requests.Session()
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Referer': 'https://map.17173.com/',
        'Origin': 'https://map.17173.com'
    }

    total_tiles = (X_MAX - X_MIN + 1) * (Y_MAX - Y_MIN + 1)
    current_tile = 0

    # 3. 循环遍历并下载拼图
    for x in range(X_MIN, X_MAX + 1):
        for y in range(Y_MIN, Y_MAX + 1):
            current_tile += 1
            url = BASE_URL.format(x=x, y=y)
            print(f"进度 {current_tile}/{total_tiles} | 正在下载瓦片: X={x}, Y={y} ...")

            try:
                response = session.get(url, headers=headers, timeout=10)

                if response.status_code == 200:
                    # 将下载的二进制数据转为图片对象
                    img_data = BytesIO(response.content)
                    tile = Image.open(img_data).convert("RGBA")

                    # 计算精确的粘贴坐标
                    paste_x = (x - X_MIN) * TILE_SIZE
                    paste_y = (y - Y_MIN) * TILE_SIZE

                    # 贴入画布
                    result_image.paste(tile, (paste_x, paste_y))
                else:
                    # 没画图的虚空边界报 404 是正常的，跳过即可
                    print(f"  [跳过] X={x}, Y={y} 处为空白区域 (状态码: {response.status_code})")

                # 延时 0.1 秒，避免请求过快被服务器拉黑
                time.sleep(0.1)

            except Exception as e:
                print(f"  [错误] 下载 X={x}, Y={y} 失败: {e}")

    # 4. 保留透明背景，以 RGBA WebP 输出（透明区域可用 alpha mask 排除干扰特征）
    print("\n[完成] 下载与拼接完成！正在压缩保存...")
    os.makedirs(OUT_DIR, exist_ok=True)
    save_path = os.path.join(OUT_DIR, "map_z7.webp")
    result_image.save(save_path, quality=90, method=6)
    print(f"[OK] 大功告成！文件已保存至: {save_path}")

    # 5. 特征分析 & 增强版输出
    analyze_features(result_image, OUT_DIR)


def analyze_features(rgba_image: Image.Image, out_dir: str) -> None:
    """
    用 SIFT 统计原始图像的特征点数量（仅诊断用，不再生成预处理副本）。
    运行时的 adaptive_clahe_map 会在加载原图后按分块自适应增强，
    因此文件级预处理无实质收益（IMREAD_GRAYSCALE 丢弃颜色，运行时 CLAHE 再次覆盖）。
    """
    try:
        import cv2
    except ImportError:
        print("  [跳过特征分析] 未安装 opencv-python，运行: pip install opencv-python")
        return

    print("\n[特征分析] 正在统计 SIFT 特征点...")
    img_np = np.array(rgba_image)            # H x W x 4, RGB 顺序
    alpha  = img_np[:, :, 3]
    gray   = cv2.cvtColor(img_np[:, :, :3], cv2.COLOR_RGB2BGR)
    gray   = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    mask   = (alpha > 0).astype(np.uint8) * 255

    sift = cv2.SIFT_create()
    kp_raw, _ = sift.detectAndCompute(gray, mask)
    print(f"  原始图像         : {len(kp_raw):>7,} 个特征点")
    print("  （运行时 adaptive_clahe_map 将在此基础上做自适应分块增强）")


if __name__ == "__main__":
    # 如果没安装 requests 和 pillow，请在终端执行 pip install requests pillow opencv-python
    download_and_stitch()
