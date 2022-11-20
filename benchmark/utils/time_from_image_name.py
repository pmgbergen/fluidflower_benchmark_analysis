"""
Reading times from images. Extremely hardcoded, but it as worked for me so far
"""


class ImageTime:
    """
    Class for reading time from images in the
    format they appear on the resfys directory.
    """

    @classmethod
    def time(self, image_title: str) -> tuple[int, int, int]:
        """
        From an image title, returns the minute, hour and date information in a tuple
        """
        date = int(image_title[4:6])
        hr = int(image_title[11:13])
        min = int(image_title[13:15])
        return (min, hr, date)

    @classmethod
    def dt(self, image_title_1: str, image_title_2: str) -> int:
        """
        From two image titles, returns the number of minutes that has passed between image 1 and image 2.
        """
        min1, hr1, date1 = self.time(image_title_1)
        min2, hr2, date2 = self.time(image_title_2)
        return 24 * 60 * (date2 - date1) + 60 * (hr2 - hr1) + (min2 - min1)
