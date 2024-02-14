import cv2
import numpy as np

# 이미지 로드
#image_path = 'examples/characters/char1/joint_overlay.png'
image_path = 'C:/Users/alpa/dongsWorks/animateDrawingsRender/AnimatedDrawings/examples/characters/charSample/texture.png'
image = cv2.imread(image_path)


# 콜백 함수 정의
def on_mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f'마우스 클릭 좌표: ({x}, {y})')

# 윈도우 생성 및 이미지 표시
cv2.namedWindow('Image')
cv2.imshow('Image', image)

# 마우스 클릭 이벤트에 콜백 함수 연결
cv2.setMouseCallback('Image', on_mouse_click)

# 아무 키나 누르면 창 종료
cv2.waitKey(0)
cv2.destroyAllWindows()
