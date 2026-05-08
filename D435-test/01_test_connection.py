"""D435 연결/펌웨어 확인. 스트림은 시작하지 않음 — 인식만 검증.

성공 시: 시리얼/펌웨어/USB 타입/지원 stream 목록 출력.
실패 시: pyrealsense2 가 device 0 개라고 답하면 USB 케이블/포트 의심.
"""
import pyrealsense2 as rs

ctx = rs.context()
devices = ctx.query_devices()
print(f"Detected {len(devices)} device(s)")

if len(devices) == 0:
    raise SystemExit("No RealSense device. USB 3.x 포트인지, 케이블이 데이터+전원 둘 다 통하는지 확인.")

for i, d in enumerate(devices):
    name = d.get_info(rs.camera_info.name)
    serial = d.get_info(rs.camera_info.serial_number)
    fw = d.get_info(rs.camera_info.firmware_version)
    usb = d.get_info(rs.camera_info.usb_type_descriptor)
    print(f"\n[{i}] {name}")
    print(f"  serial   : {serial}")
    print(f"  firmware : {fw}")
    print(f"  USB      : {usb}   (3.x 가 아니면 depth 프레임 드랍 가능)")

    print(f"  sensors  :")
    for s in d.query_sensors():
        sname = s.get_info(rs.camera_info.name)
        n_profiles = len(s.get_stream_profiles())
        print(f"    - {sname}  ({n_profiles} profiles)")

print("\nOK — 연결 정상. 다음: 02_capture_one_frame.py")
