import cv2
import numpy as np

# 비디오 파일 열기
video_capture = cv2.VideoCapture('./data/video_20240829_204726.avi')

# crop 영역 좌상단과 우하단 좌표 설정
crop_top_left = (360, 400)  # (x, y)  346 260  173
crop_bottom_right = (670, 640)  # (x, y) 540 350

# crop 영역의 너비와 높이 계산
crop_width = crop_bottom_right[0] - crop_top_left[0]
crop_height = crop_bottom_right[1] - crop_top_left[1]

gamma = 0.65  # You can adjust this value depending on the desired correction
inv_gamma = 1.0 / gamma

def equalized_frame_RGB(frame):
    r, g, b = cv2.split(frame)
    re, ge, be = cv2.equalizeHist(r), cv2.equalizeHist(g), cv2.equalizeHist(b)
    equalized_frame = cv2.merge([re, ge, be])
    return equalized_frame

def equalized_frame_grayscale(frame):
    # 프레임을 grayscale로 변환
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 히스토그램 평활화 적용
    equalized_frame = cv2.equalizeHist(gray_frame)
    return equalized_frame

def gamma_correction_frame(frame):
    table = [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)
    gamma_correction_frame = cv2.LUT(frame, table)
    return gamma_correction_frame


# 비디오 저장을 위한 설정 (crop된 크기에 맞추기)
crop_video = cv2.VideoWriter('./data/video_20240829_204726_crop.avi',
                             cv2.VideoWriter_fourcc(*'XVID'),
                             30,
                             (crop_width, crop_height),
                             isColor=True)  # grayscale 비디오를 저장하려면 isColor를 False로 설정

while True:
    # 비디오에서 프레임 읽기
    ret, frame = video_capture.read()

    if not ret:
        break  # 프레임을 더 이상 읽을 수 없으면 루프 종료

    # crop 영역만큼 프레임 crop
    cropped_frame = frame[crop_top_left[1]:crop_bottom_right[1], crop_top_left[0]:crop_bottom_right[0]]

    #view_hist_frame = equalized_frame_RGB(cropped_frame)
    #view_grayscale_frame = equalized_frame_grayscale(cropped_frame)
    #view_gamma_frame = gamma_correction_frame(cropped_frame)
    #view_gamma_hist_frame = equalized_frame_RGB(view_gamma_frame)
    #view_hist_gamma_frame = gamma_correction_frame(view_hist_frame)

    # crop된 프레임 저장
    crop_video.write(cropped_frame)

    # 원본 비디오, grayscale 영상, 평활화된 영상 함께 표시
    cv2.imshow('Cropped Video', cropped_frame)
    #cv2.imshow('Grayscale Video', view_grayscale_frame)
    #cv2.imshow('gamma Video', view_gamma_frame)
    #cv2.imshow('Equalized Video', view_hist_gamma_frame)

    # 'q'를 누르면 비디오 재생 중지 #이거 누르면 누른 만큼만 영상 저장 됨#
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
video_capture.release()
crop_video.release()
cv2.destroyAllWindows()
