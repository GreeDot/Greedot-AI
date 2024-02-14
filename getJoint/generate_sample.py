import cv2
import numpy as np

# 이미지 로드
image_path = 'examples/characters/char_sample/sample_dong.png'
image = cv2.imread(image_path)
image_resized = cv2.resize(image, (256, 256))

# 그레이스케일 변환
gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
cv2.imshow('Binary Image', gray)
cv2.waitKey(0)

# 이미지 이진화
ret, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

# 결과 이미지 저장 또는 표시
cv2.imshow('Binary Image', thresh)
cv2.imwrite('examples/characters/char_sample/mask.png', thresh)
cv2.imwrite('examples/characters/char_sample/texture.png', image_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
