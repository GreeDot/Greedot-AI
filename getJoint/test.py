import cv2

# 이미지 로드
image_path = 'examples/characters/char_sample/mask.png'
image = cv2.imread(image_path)

# 이미지 크기 측정
height, width, channels = image.shape

# 결과 출력
print(f'이미지의 높이: {height} 픽셀')
print(f'이미지의 너비: {width} 픽셀')
print(f'이미지의 채널 수: {channels}')

# 이미지 표시 (확인용)
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()