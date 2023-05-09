import asyncio
import itertools
import tkinter as tk
import traceback
from functools import partial
from pathlib import Path
from tempfile import gettempdir
from threading import Thread
from tkinter import filedialog, ttk
from typing import Optional, Callable, Any, Iterable, TypeVar

import cv2
import numpy as np
from PIL import Image, ImageTk
from fake_useragent import UserAgent
from matplotlib import pyplot as plt
from tqdm import tqdm

import streetdownloader.streetviewapi
from streetdownloader.actions import generate_all_povs, retrieve_views_from_metadata, \
    retrieve_panoramas_from_metadata, generate_metadata_from_coords
from streetdownloader.common import Location, DEFAULT_POVS_ARGS, LimitedClientSession
from streetdownloader.location import grid_coords_between, DEFAULT_METERS_BETWEEN
from streetdownloader.panoscraper import Panorama, scrape_panorama, get_panorama_photometa, DEFALUT_ZOOM_LEVEL
from streetdownloader.streetviewapi import Metadata

_T_co = TypeVar("_T_co", covariant=True)


class MapsDriver:
    def __init__(self):
        from selenium.webdriver import Chrome
        from selenium.webdriver.common.by import By
        import chromedriver_autoinstaller
        chromedriver_autoinstaller.install()

        self.driver = Chrome()
        self.driver.get('https://www.openstreetmap.org/')
        if _t := self.driver.find_elements(By.CSS_SELECTOR, 'welcome button.btn-close'):
            _t[0].click()

    def get_location(self):
        from urllib import parse
        map_fragment = dict(parse.parse_qsl(parse.urlsplit(self.driver.current_url).fragment))['map']
        _, x, y = map_fragment.split('/')
        return Location(float(x), float(y))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.driver.quit()


def user_input():
    print('Folder:')
    folder = filedialog.askdirectory(title='Select Folder')
    print(folder)

    print('Please navigate to your desired locations on the Google Maps page that has been opened in the browser.')
    print('Once you have selected your location, press enter on the terminal to continue.')
    with MapsDriver() as driver:
        input('Location 1:')
        l1 = driver.get_location()
        input('Location 2:')
        l2 = driver.get_location()
    return folder, l1, l2


class StreetDownloaderGUI:
    def __init__(self):
        self.metadata_list: dict[Metadata] = {}

        self.root = tk.Tk()
        self.root.title('MyApp')
        self.root.geometry("640x360")
        self.cache_folder = Path(gettempdir()) / 'street-downloader'
        self.cache_folder.mkdir(exist_ok=True)
        self.active_progress = False
        self.driver: Optional[MapsDriver] = None

        self.header = ttk.Frame(self.root)
        self.header.pack()
        self.header_home = ttk.Button(self.header, text='Home', command=self.show_home_frame)
        self.header_home.grid(row=0, column=0)
        self.header_metadata = ttk.Button(self.header, text='Metadata', command=self.show_metadata_frame)
        self.header_metadata.grid(row=0, column=1)
        self.header_view = ttk.Button(self.header, text='View', command=self.show_view_frame)
        self.header_view.grid(row=0, column=2)
        self.header_panorama = ttk.Button(self.header, text='Panorama', command=self.show_panorama_frame)
        self.header_panorama.grid(row=0, column=3)
        self.header_buttons = self.header_home, self.header_metadata, self.header_view, self.header_panorama

        self.home_frame = ttk.LabelFrame(self.root)
        self.metadata_frame = ImageFrame(self.root)
        self.view_frame = ImageFrame(self.root)
        self.panorama_frame = ImageFrame(self.root)

        if self.home_frame:
            folder_row = ttk.Frame(self.home_frame)
            folder_row.pack(pady=5)
            ttk.Label(folder_row, text="Folder:").pack(side=tk.LEFT, padx=5)
            self.folder_entry = ttk.Entry(folder_row, width=30)
            self.folder_entry.pack(side=tk.LEFT, padx=5)
            self.folder_button = ttk.Button(folder_row, text="Select", command=self.ask_user_path)
            self.folder_button.pack(side=tk.LEFT, padx=5)

            location_frame = ttk.Frame(self.home_frame)
            location_frame.pack(pady=5)
            self.coord1_row = LabelLocation(location_frame, "Location 1:")
            self.coord1_row.pack()
            self.coord2_row = LabelLocation(location_frame, "Location 2:")
            self.coord2_row.pack()
            driver_frame = ttk.Frame(location_frame)
            driver_frame.pack()
            self.driver_button = ttk.Button(driver_frame, text="Start Browser", command=self.toggle_driver)
            self.driver_button.pack(side=tk.LEFT, padx=5)
            self.coord1_button = ttk.Button(driver_frame, text="Set Loc1", state=tk.DISABLED,
                                            command=partial(self.location_from_driver, loc1=True))
            self.coord1_button.pack(side=tk.LEFT, padx=5)
            self.coord2_button = ttk.Button(driver_frame, text="Set Loc2", state=tk.DISABLED,
                                            command=partial(self.location_from_driver, loc2=True))
            self.coord2_button.pack(side=tk.LEFT, padx=5)

            metadata_frame = ttk.Frame(self.home_frame)
            metadata_frame.pack(pady=15)
            self.metadata_progress = LabelProgress(metadata_frame, "Start metadata research")
            self.metadata_progress.pack(side=tk.LEFT, padx=5)
            metadata_buttons = ttk.Frame(metadata_frame)
            metadata_buttons.pack(side=tk.LEFT, padx=5)
            self.metadata_button = ttk.Button(metadata_buttons, text="Get Metadata",
                                              command=self.get_metadata_with_progress)
            self.metadata_button.pack()
            self.metadata_export = ttk.Button(metadata_buttons, text="Export Metadata", state="disabled",
                                              command=self.export_metadata)
            self.metadata_export.pack()

            images_frame = ttk.Frame(self.home_frame)
            images_frame.pack(pady=15)
            self.images_progress = LabelProgress(images_frame, "Start images download")
            self.images_progress.pack(side=tk.LEFT, padx=5)
            view_panorama_buttons = ttk.Frame(images_frame)
            view_panorama_buttons.pack(pady=5)
            self.view_button = ttk.Button(view_panorama_buttons, text="Get Views", state="disabled",
                                          command=partial(self.get_images_with_progress, view=True))
            self.view_button.pack()
            self.panorama_button = ttk.Button(view_panorama_buttons, text="Get Panoramas", state="disabled",
                                              command=partial(self.get_images_with_progress, panorama=True))
            self.panorama_button.pack()

            self.skip_checkbox = LabelCheck(view_panorama_buttons, text="Skip existing", value=True)
            self.skip_checkbox.pack()

        if self.metadata_frame:
            api_key_row = ttk.Frame(self.metadata_frame.content)
            api_key_row.pack(pady=5)
            ttk.Label(api_key_row, text="Google API key").pack(side=tk.LEFT, padx=5)
            self.api_key_var = tk.StringVar(value=streetdownloader.streetviewapi.GOOGLE_API_KEY)
            self.api_key_entry = ttk.Entry(api_key_row, show="*", textvariable=self.api_key_var)
            self.api_key_entry.pack(side=tk.LEFT, padx=5)

            self.meters_grid_slider = LabelScale(self.metadata_frame.content, 'Meters', 1, 20,
                                                 start=DEFAULT_METERS_BETWEEN, command=self.update_metadata_image)
            self.meters_grid_slider.pack()

            self.google_checkbox = LabelCheck(self.metadata_frame.content, text="Google only", value=True)
            self.google_checkbox.pack(pady=5)

        if self.view_frame:
            self.povs = DEFAULT_POVS_ARGS
            self.angle_sliders = ScaleList(self.view_frame.content, 1, 30, [(f"{g}", n) for n, g in self.povs],
                                           command=self.update_view_image)
            self.angle_sliders.pack(pady=5)
            self.crop_checkbox = LabelCheck(self.view_frame.content, text="Crop Image",
                                            value=streetdownloader.streetviewapi.CROP_GOOGLE_LOGO)
            self.crop_checkbox.pack(pady=5)

        if self.panorama_frame:
            self.zoom_slider = LabelScale(self.panorama_frame.content, "Zoom", 1, 5, start=DEFALUT_ZOOM_LEVEL,
                                          command=self.update_panorama_image)
            self.zoom_slider.pack(pady=5)
            zoom_info_frame = ttk.Frame(self.panorama_frame.content)
            zoom_info_frame.pack(pady=5)
            self.zoom_grid_label = ttk.Label(zoom_info_frame, text='Grid size:')
            self.zoom_grid_label.pack(side=tk.LEFT, padx=5)
            self.zoom_total_label = ttk.Label(zoom_info_frame, text='Total images:')
            self.zoom_total_label.pack(side=tk.LEFT, padx=5)
            self.pano_size_label = ttk.Label(self.panorama_frame.content, text='Panorama resolution:')
            self.pano_size_label.pack(pady=5)

        self.show_home_frame()

        Thread(target=self._init_cache_panorama).start()
        Thread(target=self._init_cache_metadata).start()

    def run(self, n=0):
        return self.root.mainloop(n)

    def show_home_frame(self):
        self.hide_all_frames()
        self.home_frame.pack(expand=tk.TRUE, fill=tk.BOTH)

    def show_metadata_frame(self):
        self.hide_all_frames()
        self.metadata_frame.pack(expand=tk.TRUE, fill=tk.BOTH)

    def show_view_frame(self):
        self.hide_all_frames()
        self.view_frame.pack(expand=tk.TRUE, fill=tk.BOTH)
        self.update_view_image(self.angle_sliders.get())

    def show_panorama_frame(self):
        self.hide_all_frames()
        self.panorama_frame.pack(expand=tk.TRUE, fill=tk.BOTH)
        self.update_panorama_image(self.zoom_slider.var.get())

    def hide_all_frames(self):
        self.home_frame.pack_forget()
        self.metadata_frame.pack_forget()
        self.view_frame.pack_forget()
        self.panorama_frame.pack_forget()

    def ask_user_path(self):
        folder = filedialog.askdirectory(title='Select Folder')
        self.folder_entry.delete(0, tk.END)
        self.folder_entry.insert(0, folder)

    def stop_progress(self):
        self.active_progress = False

    def _init_cache_panorama(self):
        panorama_path = self.cache_folder / f'panorama.jpg'
        if panorama_path.exists():
            return self.update_panorama_image(self.zoom_slider.var.get())

        async def _f():
            pano_id = '1irPm5Sn3Aemjr2oh9H0dg'
            async with LimitedClientSession(5, 1) as limited_session:
                pm = await get_panorama_photometa(limited_session, pano_id)
                panorama = Panorama(photometa=pm)
                await scrape_panorama(limited_session, panorama, 1)
            panorama.save(panorama_path, True)
            self.update_panorama_image(self.zoom_slider.var.get())

        asyncio.run(_f())

    def _init_cache_metadata(self):
        metadata_img = self.cache_folder / f'metadata.jpg'
        if metadata_img.exists():
            return self.update_metadata_image(self.meters_grid_slider.var.get())

        async def get_tile(session, map_type, zoom, x_tile, y_tile):
            base_url = "http" + "://tile.stamen.com/{}/{}/{}/{}.png"
            url = base_url.format(map_type, zoom, x_tile, y_tile)
            async with session.get(url) as response:
                buffer = await response.read()
            img_np = cv2.imdecode(np.frombuffer(buffer, dtype=np.uint8), cv2.IMREAD_COLOR)
            return img_np

        async def _f():
            start_x, start_y = 140139, 97406
            k_x, k_y = 6, 4
            tiles_coords_list = tuple((i, j) for j in range(k_y) for i in range(k_x))
            async with LimitedClientSession(50, 10) as session:
                session.headers.update({'User-Agent': UserAgent().random})
                coro = [get_tile(session, 'terrain', 18, start_x + x, start_y + y) for x, y in tiles_coords_list]
                tiles = await asyncio.gather(*coro)
            tile_shape = max(tile.shape for tile in tiles if tile is not None)
            reshaped = np.zeros((k_y, k_x, *tile_shape))
            tiles_with_coords = tuple(zip(tiles, tiles_coords_list))
            for tile, coords in tiles_with_coords:
                i, j = coords
                reshaped[j, i] = tile
            img = np.hstack(np.hstack(reshaped))
            cv2.imwrite(str(metadata_img), img)
            self.update_metadata_image(self.meters_grid_slider.var.get())

        asyncio.run(_f())

    def toggle_driver(self):
        if self.driver is None:
            self.driver = MapsDriver()
            self.driver_button.config(text='Stop Browser')
            self.coord1_button.config(state=tk.NORMAL)
            self.coord2_button.config(state=tk.NORMAL)
        else:
            self.driver.driver.quit()
            self.driver = None
            self.driver_button.config(text='Start Browser')
            self.coord1_button.config(state=tk.DISABLED)
            self.coord2_button.config(state=tk.DISABLED)

    def location_from_driver(self, loc1=False, loc2=False):
        from selenium.common import NoSuchWindowException
        try:
            loc = self.driver.get_location()
        except NoSuchWindowException:
            return self.toggle_driver()
        if loc1:
            self.coord1_row.set(loc)
        elif loc2:
            self.coord2_row.set(loc)

    async def _get_metadata_with_progress(self):
        loc1 = self.coord1_row.get()
        loc2 = self.coord2_row.get()
        mb = self.meters_grid_slider.var.get()
        from_google = self.google_checkbox.var.get()
        generator, total = grid_coords_between(loc1, loc2, mb)
        metadata_list = []
        with self.metadata_progress.create_context(total) as t:
            async for metadata in generate_metadata_from_coords(generator, from_google, True):
                if not self.active_progress:
                    break
                if metadata:
                    metadata_list.append(metadata)
                t.update(1)
                self.metadata_progress.update_progress()
        self.metadata_list = metadata_list

    def get_metadata_with_progress(self):
        loc1 = self.coord1_row.get()
        loc2 = self.coord2_row.get()
        api_key = self.api_key_entry.get()
        if loc1 is None or loc2 is None or not api_key:
            return
        streetdownloader.streetviewapi.GOOGLE_API_KEY = api_key
        self.metadata_progress.clear_progress()
        self.metadata_button.config(command=self.stop_progress, text="Stop Metadata")
        buttons = self.metadata_export, self.view_button, self.panorama_button
        for button in itertools.chain(buttons, self.header_buttons):
            button.config(state=tk.DISABLED)
        self.active_progress = True

        asyncio.run(self._get_metadata_with_progress())
        self.active_progress = False
        self.metadata_progress.bar['value'] = 100
        self.metadata_progress.bar.update()
        k = len(self.metadata_list)
        self.metadata_progress.label.config(text=f"{k} location{'' if k == 1 else 's'} found")
        self.metadata_progress.label.update()
        self.metadata_button.config(command=self.get_metadata_with_progress, text="Get Metadata")
        for button in self.header_buttons:
            button.config(state=tk.NORMAL)
        if k:
            for button in buttons:
                button.config(state=tk.NORMAL)

    async def _get_views_with_progress(self):
        skip_existing = self.skip_checkbox.var.get()
        folder = Path(self.folder_entry.get())
        crop_logo = self.crop_checkbox.var.get()
        with self.images_progress.create_context(len(self.metadata_list) * sum(x[0] for x in self.povs)) as t:
            async for view in retrieve_views_from_metadata(self.metadata_list, folder, self.povs, False, skip_existing):
                if not self.active_progress:
                    break
                view.crop_logo = crop_logo
                view.save(folder, True, True)
                t.update(1)
                self.images_progress.update_progress()

    async def _get_panoramas_with_progress(self):
        skip_existing = self.skip_checkbox.var.get()
        folder = Path(self.folder_entry.get())
        zoom = self.zoom_slider.var.get()
        with self.images_progress.create_context(len(self.metadata_list)) as t:
            async for panorama in retrieve_panoramas_from_metadata(
                    self.metadata_list, folder, zoom, skip_existing, False):
                if not self.active_progress:
                    break
                panorama.save(folder, True, True)
                t.update(1)
                self.images_progress.update_progress()

    def get_images_with_progress(self, panorama=False, view=False):
        if not self.metadata_list or not self.folder_entry.get():
            return
        self.images_progress.clear_progress()
        if view:
            coroutine = self._get_views_with_progress()
            action_button, *other_buttons = self.view_button, self.panorama_button, self.metadata_button
        elif panorama:
            coroutine = self._get_panoramas_with_progress()
            action_button, *other_buttons = self.panorama_button, self.view_button, self.metadata_button
        else:
            return
        original_text = action_button.cget('text')
        original_command = partial(self.get_images_with_progress, view=view, panorama=panorama)
        other_buttons = *other_buttons, *self.header_buttons
        action_button.config(command=self.stop_progress, text="Stop Download")
        for button in other_buttons:
            button.config(state=tk.DISABLED)
        self.active_progress = True
        asyncio.run(coroutine)
        self.active_progress = False
        for button in other_buttons:
            button.config(state=tk.NORMAL)
        action_button.config(command=original_command, text=original_text)
        self.images_progress.bar['value'] = 100
        self.images_progress.bar.update()
        self.images_progress.label.config(text=f"Download completed")
        self.images_progress.label.update()

    def export_metadata(self):
        file_path = Path(filedialog.asksaveasfilename(
            defaultextension='.csv', initialfile='metadata',
            filetypes=(('Comma Separated Values', '*.csv'), ('JSON Serialized Values', '*.json'),)
        ))
        with open(file_path, mode='w', newline='', encoding='utf-8') as fp:
            if file_path.suffix == '.csv':
                import csv
                fieldnames = 'status', 'copyright', 'date', 'lat', 'lng', 'pano_id'
                writer = csv.DictWriter(fp, fieldnames=fieldnames)
                writer.writeheader()
                for metadata in tqdm(self.metadata_list):
                    writer.writerow({
                        'status': metadata.status.value,
                        'copyright': metadata.copyright,
                        'date': metadata.date,
                        'lat': metadata.location.lat,
                        'lng': metadata.location.lng,
                        'pano_id': metadata.pano_id
                    })
            elif file_path.suffix == '.json':
                import json
                import cattrs
                fp.write('[\n')
                for i, meta in enumerate(tqdm(self.metadata_list)):
                    if i > 0:
                        fp.write(',\n')
                    fp.write(json.dumps(cattrs.unstructure(meta), indent=3))
                fp.write('\n]\n')

    def update_metadata_image(self, value: int):
        new_image_path = self.cache_folder / f'metadata_{value}.jpg'
        if not new_image_path.exists():
            panorama_path = self.cache_folder / f'metadata.jpg'
            if not panorama_path.exists():
                return
            img = cv2.imread(str(panorama_path))
            top_left_img = Location(41.9045, 12.4514)
            bottom_right_img = Location(41.9003, 12.4605)
            coords, _ = grid_coords_between(Location(41.904, 12.4524), Location(41.901, 12.4591), value)
            scale_lat = (bottom_right_img.lat - top_left_img.lat) / img.shape[0]
            scale_lng = (bottom_right_img.lng - top_left_img.lng) / img.shape[1]
            size = 2 + value >> 2
            for coord in coords:
                x = int((coord.lng - top_left_img.lng) / scale_lng)
                y = int((coord.lat - top_left_img.lat) / scale_lat)
                cv2.circle(img, (x, y), size, (255, 127, 0), thickness=-1)
            cv2.imwrite(str(new_image_path), img, [cv2.IMWRITE_JPEG_QUALITY, 50])

        self.metadata_frame.update_image(new_image_path)

    def update_panorama_image(self, value: int):
        new_image_path = self.cache_folder / f'panorama_zoom_{value}.jpg'
        if not new_image_path.exists():
            panorama_path = self.cache_folder / f'panorama.jpg'
            if not panorama_path.exists():
                return
            img = cv2.imread(str(panorama_path))

            def apply_lines(image, n):
                if n == 0:
                    return img

                color = np.array([0, 255, 0]) * (.2 + .15 * n)
                color = tuple(map(int, color))
                height, width, _ = img.shape
                segment_width = width >> n
                segment_height = height >> (n - 1)

                line_thickness = max(1, int((img.shape[0] / 60) * (0.5 ** (n - 1))))
                for i in range(1, 2 ** n):
                    start_point = (i * segment_width, 0)
                    end_point = (i * segment_width, height)
                    image = cv2.line(image, start_point, end_point, color, line_thickness)

                for i in range(1, 2 ** (n - 1)):
                    start_point = (0, segment_height * i)
                    end_point = (width, segment_height * i)
                    image = cv2.line(image, start_point, end_point, color, line_thickness)

                return apply_lines(image, n - 1)

            cv2.imwrite(str(new_image_path), apply_lines(img, value), [cv2.IMWRITE_JPEG_QUALITY, 50])

        self.panorama_frame.update_image(new_image_path)
        self.zoom_grid_label.config(text=f"Grid size: {2 ** value}x{2 ** (value - 1)}")
        self.zoom_total_label.config(text=f"Total images: {2 ** (2 * value - 1)}")
        self.pano_size_label.config(text=f"Panorama resolution: {2 ** (value + 9)}x{2 ** (value + 8)}")

    def update_view_image(self, values: Iterable[tuple[str, int]]):
        self.povs = sorted(map(lambda t: (t[1], int(''.join(filter(str.isdigit, t[0])))), values), key=lambda x: x[1])
        path = self.cache_folder / ('_'.join(['view'] + ['-'.join(map(str, d)) for d in self.povs]) + '.jpg')
        if not path.exists():
            from matplotlib.colors import hsv_to_rgb
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            for heading, fov, pitch in generate_all_povs(*self.povs):
                heading_rad, fov_rad, pitch_rad = np.radians(heading), np.radians(fov), np.radians(pitch)
                center = np.array([np.cos(heading_rad) * np.cos(pitch_rad),
                                   np.sin(heading_rad) * np.cos(pitch_rad),
                                   np.sin(pitch_rad)])
                half_side_length = np.tan(fov_rad / 2)

                side1 = np.cross(center, [0, 0, 1])
                side2 = np.cross(center, side1)

                side1 = side1 / np.linalg.norm(side1) * half_side_length
                side2 = side2 / np.linalg.norm(side2) * half_side_length

                square = np.array([- side1 - side2, side1 - side2, side2 + side1, side2 - side1]) + center

                h = (437 * (heading + pitch) * np.pi) % 1
                p = (pitch / 90) ** 2 * (823 * pitch * np.pi) % 1

                color = hsv_to_rgb(np.array([h, 1 - p, 1]))
                sq = Poly3DCollection([square], alpha=0.3 + pitch / 256, facecolors=color)
                ax.add_collection3d(sq)

            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([-1, 1])

            ax.set_axis_off()

            plt.savefig(path)
            plt.close(fig)
        self.view_frame.update_image(path)


class LabelProgress(ttk.Frame):
    def __init__(self, master, init_text=''):
        super().__init__(master)
        self.label = ttk.Label(self, text=init_text)
        self.label.pack()
        self.bar = ttk.Progressbar(self, orient="horizontal", length=200, mode="determinate")
        self.bar.pack()
        self.tqdm = None

    def create_context(self, total: int):
        self.tqdm = tqdm(ncols=0, total=total, smoothing=0.01)
        return self.tqdm

    def update_progress(self):
        if not self.tqdm:
            return
        self.label.config(text=str(self.tqdm))
        self.label.update()
        self.bar['value'] = 100 * self.tqdm.n / self.tqdm.total
        self.bar.update()

    def clear_progress(self):
        self.bar['value'] = 0
        self.bar.update()
        self.label.config(text='')
        self.bar.update()


class LabelLocation(ttk.Frame):
    def __init__(self, master, text=''):
        super().__init__(master)
        self.label = ttk.Label(self, text=text)
        self.validate_command = (self.register(self._validate_float), '%P')
        self.lat_var = tk.DoubleVar()
        self.lng_var = tk.DoubleVar()
        self.lat_entry = ttk.Entry(self, validate='key', validatecommand=self.validate_command,
                                   textvariable=self.lat_var, width=12)
        self.lng_entry = ttk.Entry(self, validate='key', validatecommand=self.validate_command,
                                   textvariable=self.lng_var, width=12)
        self.paste = ttk.Button(self, text='Paste', command=self.paste, width=5)

        self.label.grid(row=0, column=0, padx=5)
        self.lat_entry.grid(row=0, column=1, padx=5)
        self.lng_entry.grid(row=0, column=2, padx=5)
        self.paste.grid(row=0, column=3, padx=5)

    @staticmethod
    def _validate_float(value):
        try:
            if value:
                float(value)
            return True
        except ValueError:
            return False

    def get(self) -> Location:
        return Location(lat=float(self.lat_entry.get()), lng=float(self.lng_entry.get()))

    def set(self, location: Location):
        self.lat_var.set(location.lat)
        self.lng_var.set(location.lng)

    def paste(self, *_):
        import pyperclip
        clip = pyperclip.paste()
        try:
            lat_str, lng_str, *_ = clip.split(',')
            self.set(Location(float(lat_str), float(lng_str)))
        except ValueError:
            traceback.print_exc()


class LabelCheck(ttk.Frame):
    def __init__(self, master, text: str, value: bool = False, command: Optional[Callable[[bool], Any]] = None):
        super().__init__(master)

        self.text = text
        self.var = tk.BooleanVar(value=value)
        self.command = command
        self.button = tk.Checkbutton(self, variable=self.var, command=self.wrapped_command)
        self.button.pack(side=tk.LEFT)
        self.label = ttk.Label(self, text=text)
        self.label.pack(side=tk.LEFT)

    def wrapped_command(self, *_):
        if self.command:
            self.command(self.var.get())


class LabelScale(ttk.Frame):
    def __init__(self, parent, text: str, from_: int, to: int, start: Optional[int] = None,
                 command: Optional[Callable[[int], Any]] = None, **kwargs):
        super().__init__(parent)

        self.text = text
        self.label = ttk.Label(self)
        self.label.pack(side=tk.LEFT)
        self.command = command

        self.var = tk.IntVar()
        self.scale = tk.Scale(self, orient=tk.HORIZONTAL, variable=self.var, from_=from_, to=to,
                              command=self.wrapped_command, showvalue=False, **kwargs)
        if start is not None:
            self.var.set(start)
            self.scale.set(start)
        self.scale.pack(side=tk.RIGHT, padx=5, pady=5)
        self.label.configure(text=f"{self.text}: {self.var.get()}")

    def wrapped_command(self, *_):
        self.label.configure(text=f"{self.text}: {self.var.get()}")
        if self.command:
            return self.command(self.var.get())


class ScaleList(ttk.LabelFrame):
    def __init__(self, parent, from_: int, to: int, items: Iterable[tuple[str, int]],
                 command: Optional[Callable[[Iterable[tuple[str, int]]], Any]] = None,
                 sanitiser: Optional[Callable[[str], str]] = None):
        super().__init__(parent)

        self.scales = []
        self.from_ = from_
        self.to_ = to
        self.command = command
        self.sanitiser = sanitiser

        self.scale_container = ttk.Frame(self)
        self.scale_container.pack(fill=tk.BOTH, expand=tk.TRUE)

        for name, value in items:
            self.add_scale(name, value)

        self.entry = ttk.Entry(self)
        self.entry.pack(side=tk.LEFT, padx=5)

        self.add_button = ttk.Button(self, text="+", command=self.add_entry_scale)
        self.add_button.pack(side=tk.LEFT, padx=5)

    def add_scale(self, name, value):
        def delete_scale():
            for item in (scale, delete_button, frame):
                item.pack_forget()
                item.destroy()
            self.scales.remove(scale)
            self.wrapped_command()

        frame = tk.Frame(self.scale_container)
        scale = LabelScale(frame, name, self.from_, self.to_, value, command=self.wrapped_command)
        scale.pack(side=tk.LEFT, fill=tk.X)

        delete_button = ttk.Button(frame, text="-", command=delete_scale)
        delete_button.pack(side=tk.RIGHT)
        frame.pack(fill=tk.X)

        self.scales.append(scale)

    def add_entry_scale(self):
        name = self.entry.get().strip()
        if self.sanitiser is not None:
            name = self.sanitiser(name)
        if name:
            self.add_scale(name, self.from_)
            self.entry.delete(0, tk.END)
            self.wrapped_command()

    def get(self):
        return [(scale.text, scale.var.get()) for scale in self.scales]

    def wrapped_command(self, *_):
        if self.command:
            self.command(self.get())


class ImageFrame(ttk.LabelFrame):
    def __init__(self, master, image_path=None):
        super().__init__(master)

        self.tk_image = None
        self.image = None
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)

        self.image_label = ttk.Label(self)
        self.image_label.grid(row=0, column=0, sticky="ew")
        self.grid_columnconfigure(1, weight=1, uniform="kkk")

        self.content = ttk.Frame(self)
        self.content.grid(row=0, column=1, sticky="ew")
        self.grid_columnconfigure(1, weight=1, uniform="kkk")

        self.image_label.bind("<Configure>", self.resize_image)
        self.update_image(image_path)

    def update_image(self, image_path: Optional[Path]):
        try:
            self.image = Image.open(image_path)
            self.resize_image()
        except (AttributeError, FileNotFoundError, ValueError):
            self.image_label.config(image="")
            self.image_label.image = None
            self.image_label.config(text=' ')

    def resize_image(self, _=None):
        if not self.image:
            return

        target_width = min(self.image_label.winfo_width(), self.master.winfo_width() >> 1)
        target_height = self.image_label.winfo_height()
        img_ratio = float(self.image.width) / float(self.image.height)
        target_ratio = float(target_width) / float(target_height)

        if target_ratio > img_ratio:
            target_width = int(target_height * img_ratio)
        else:
            target_height = int(target_width / img_ratio)
        target_size = target_width, target_height
        resized_image = self.image.resize(target_size)
        self.tk_image = ImageTk.PhotoImage(resized_image)
        self.image_label.config(image=self.tk_image)
        self.image_label.image = self.tk_image


def main():
    cache_dir = Path(gettempdir()) / 'street-downloader'
    if cache_dir.exists():
        for img in cache_dir.glob("metadata_*.jpg"):
            img.unlink(missing_ok=True)
        pass
        # rmtree(cache_dir, True)
    app = StreetDownloaderGUI()
    app.coord1_row.set(Location(41.90764, 12.4454))
    app.coord2_row.set(Location(41.90102, 12.45878))
    app.root.mainloop()


if __name__ == '__main__':
    StreetDownloaderGUI().run()
