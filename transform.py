from ImageProcessing import Image2


def brighten(image, factor):

    x_pixels, y_pixels, num_channels = image.array.shape
    new_im = Image2(x_pixels=x_pixels, y_pixels=y_pixels,
                   num_channels=num_channels)

    new_im.array = image.array * factor

    return new_im


def adjust_contrast(image, factor, mid):

    x_pixels, y_pixels, num_channels = image.array.shape
    new_im = Image2(x_pixels=x_pixels, y_pixels=y_pixels,
                   num_channels=num_channels)


    new_im.array = (image.array - mid) * factor + mid

    return new_im


def blur(image, kernel_size):

    x_pixels, y_pixels, num_channels = image.array.shape
    new_im = Image2(x_pixels=x_pixels, y_pixels=y_pixels,
                   num_channels=num_channels)
    neighbor_range = kernel_size // 2  # this is a variable that tells us how many neighbors we actually look at (ie for a kernel of 3, this value should be 1)
    for x in range(x_pixels):
        for y in range(y_pixels):
            for c in range(num_channels):
                total = 0
                for x_i in range(max(0, x - neighbor_range), min(new_im.x_pixels - 1, x + neighbor_range) + 1):
                    for y_i in range(max(0, y - neighbor_range), min(new_im.y_pixels - 1, y + neighbor_range) + 1):
                        total += image.array[x_i, y_i, c]
                new_im.array[x, y, c] = total / (kernel_size ** 2)
    return new_im


def apply_kernel(image, kernel):

    x_pixels, y_pixels, num_channels = image.array.shape
    new_im = Image2(x_pixels=x_pixels, y_pixels=y_pixels,
                   num_channels=num_channels)
    neighbor_range = kernel.shape[
                         0] // 2  # this is a variable that tells us how many neighbors we actually look at (ie for a 3x3 kernel, this value should be 1)

    for x in range(x_pixels):
        for y in range(y_pixels):
            for c in range(num_channels):
                total = 0
                for x_i in range(max(0, x - neighbor_range), min(new_im.x_pixels - 1, x + neighbor_range) + 1):
                    for y_i in range(max(0, y - neighbor_range), min(new_im.y_pixels - 1, y + neighbor_range) + 1):
                        x_k = x_i + neighbor_range - x
                        y_k = y_i + neighbor_range - y
                        kernel_val = kernel[x_k, y_k]
                        total += image.array[x_i, y_i, c] * kernel_val
                new_im.array[x, y, c] = total
    return new_im

