from fastapi import FastAPI, Query, Path, Response, HTTPException
import cv2
import base64

# please excuse my junk i haven't really written fastapi before lmfao

app = FastAPI()

videopath = "badapple-h264.mp4"
cap = cv2.VideoCapture(videopath)

@app.get("/frame/{frame_number}")
def get_bad_apple_frame(
    frame_number: int = Path(..., description="the frame u want to extract"),
    width: int = Query(640, gt=0),
    height: int = Query(480, gt=0),
    img_format: str = Query("png", regex="^(png|jpg|jpeg|bmp|json)$")
):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    ret, frame = cap.read()

    if not ret:
        return HTTPException(404)

    resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST)

    # JSON
    if img_format == "json":
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        pixels = (binary > 127).tolist()
        
        return {
            "frame_number": frame_number,
            "width": width,
            "height": height,
            "format": "json",
            "pixels": pixels
        }

    success, encoded_img = cv2.imencode(f".{img_format}", resized)
    
    # image
    mime_type = f"image/{'jpeg' if img_format in ['jpg', 'jpeg'] else 'png'}"
    return Response(content=encoded_img.tobytes(), media_type=mime_type)

@app.get("/")
def root():
    return Response(content=b"yo!! try out /frame/{frame_number}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
