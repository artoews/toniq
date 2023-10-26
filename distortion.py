
def net_pixel_bandwidth(pixel_bandwidth_2, pixel_bandwidth_1):
    if pixel_bandwidth_1 == 0:
        return pixel_bandwidth_2
    else:
        return 1 / (1 / pixel_bandwidth_2 - 1 / pixel_bandwidth_1)


