#!python3


from rich import print


def main():
    import uvc

    for device in uvc.device_list():

        cap = uvc.Capture(device["uid"])

        available_modes = cap.available_modes
        cap.close()

        for mode in available_modes:
            cap = uvc.Capture(device["uid"])
            print(f"{cap.name} running at {mode}")
            try:
                cap.frame_mode = mode
            except uvc.InitError as err:
                print(f"{cap.name} mode selection - {err}")
                continue
            try:
                for x in range(10):
                    frame = cap.get_frame_robust()
                    print("frame gray mean", frame.gray.mean())
                # print(frame.img.mean())
            except uvc.InitError as err:
                print(f"{cap.name} getting frames - {err}")

            cap.close()

    # Uncomment the following lines to configure the Pupil 200Hz IR cameras:
    # controls_dict = dict([(c.display_name, c) for c in cap.controls])
    # controls_dict['Auto Exposure Mode'].value = 1
    # controls_dict['Gamma'].value = 200

    # cap.frame_mode = cap.avaible_modes[0]
    # for x in range(100):
    #     frame = cap.get_frame_robust()
    #     print(frame.img.mean())
    # cap = None


if __name__ == "__main__":
    main()
