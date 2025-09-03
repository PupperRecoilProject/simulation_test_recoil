import asyncio
from aiohttp import web
import logging
import cv2
import time
import PIL.Image
from typing import List
import weakref

# 假設 nanoowl 相關模組可以被正確導入
from nanoowl.tree import Tree
from nanoowl.tree_predictor import TreePredictor
from nanoowl.owl_predictor import OwlPredictor
from nanoowl.tree_drawing import draw_tree_output

# --- 伺服器設定 ---
HOST = "0.0.0.0"
PORT = 8081 # 使用一個獨立的埠號
IMAGE_QUALITY = 75
RESOLUTION = (640, 480)

# --- NanoOwl 模型設定 ---
IMAGE_ENCODER_ENGINE = "/home/pupper/vlm/nanoowl/data/owl_image_encoder_patch32.engine"

# 全域變數，用於儲存 prompt 資訊
prompt_data = None

def get_csi_pipeline(capture_width, capture_height, framerate=30):
    return (
       f"nvarguscamerasrc tnr-mode=2 ! "
       f"video/x-raw(memory:NVMM), width={capture_width}, height={capture_height}, framerate={framerate}/1 ! "
       f"nvvidconv flip-method=0 ! "
       f"video/x-raw, width={capture_width}, height={capture_height}, format=(string)BGRx ! "
       f"videoconvert ! "
       f"video/x-raw, format=(string)BGR ! appsink"
    )

async def handle_index_get(request: web.Request):
    logging.info("Serving video_stream.html")
    return web.FileResponse("./video_stream.html")

async def websocket_handler(request):
    global prompt_data
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    logging.info("WebSocket client connected.")
    request.app['websockets'].add(ws)

    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                logging.info(f"Received prompt: {msg.data}")
                try:
                    tree = Tree.from_prompt(msg.data)
                    predictor = request.app['predictor']
                    clip_encodings = predictor.encode_clip_text(tree)
                    owl_encodings = predictor.encode_owl_text(tree)
                    prompt_data = {
                        "tree": tree,
                        "clip_encodings": clip_encodings,
                        "owl_encodings": owl_encodings
                    }
                except Exception as e:
                    logging.error(f"Error processing prompt: {e}")
    finally:
        request.app['websockets'].discard(ws)
    return ws

async def detection_loop(app: web.Application):
    loop = asyncio.get_running_loop()
    predictor = app['predictor']

    logging.info("Opening camera...")
    pipeline = get_csi_pipeline(RESOLUTION[0], RESOLUTION[1])
    camera = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not camera.isOpened():
        logging.error("Could not open camera.")
        return

    logging.info("Camera warming up...")
    await asyncio.sleep(2.0)
    
    def _read_process_encode():
        re, image = camera.read()
        if not re: return None

        if prompt_data:
            try:
                image_pil = PIL.Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                detections = predictor.predict(
                    image_pil,
                    tree=prompt_data['tree'],
                    clip_text_encodings=prompt_data['clip_encodings'],
                    owl_text_encodings=prompt_data['owl_encodings']
                )
                image = draw_tree_output(image, detections, prompt_data['tree'])
            except Exception:
                pass # 忽略推論錯誤

        re, jpeg_buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, IMAGE_QUALITY])
        return jpeg_buffer.tobytes() if re else None

    while True:
        image_bytes = await loop.run_in_executor(None, _read_process_encode)
        if image_bytes:
            for ws in app["websockets"]:
                await ws.send_bytes(image_bytes)
        await asyncio.sleep(0.01) # 稍微釋放CPU

    camera.release()
    
async def start_background_tasks(app):
    app['detection_task'] = asyncio.create_task(detection_loop(app))
    yield
    app['detection_task'].cancel()
    await app['detection_task']

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = web.Application()
    app['websockets'] = weakref.WeakSet()
    
    logging.info("Initializing predictor...")
    app['predictor'] = TreePredictor(
        owl_predictor=OwlPredictor(image_encoder_engine=IMAGE_ENCODER_ENGINE)
    )
    
    app.router.add_get("/", handle_index_get)
    app.router.add_get("/ws", websocket_handler)
    app.cleanup_ctx.append(start_background_tasks)
    
    logging.info(f"Starting video server on http://{HOST}:{PORT}")
    web.run_app(app, host=HOST, port=PORT)
