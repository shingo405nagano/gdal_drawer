# **gdal_drawer**
`gdal_drawer`では`gdal`のラッパーを提供しています。これは"DTM"などのシングルバンドラスターを基準に作成したので、マルチバンドラスターでは上手く動作しない場合があります。その内時間があるときに...

```python
>>> from gdal_drawer.custom import gdal_open
```
<br>


## **1. Examples of IO:**
### 1-1. RasterData の読み込み
`gdal_open`関数を使用してラスターデータを読み込みます。これは`gdal.Open`関数と同じですが、ラッパークラスの`CustomGdalDataset`を返します。
```python
>>> file_path = r'.\sample\raster.tif'
>>> dst = gdal_open(file_path)
>>> type(dst)
CustomGdalDataset
```

### 1-2. RasterData の配列を取得
これは `gdal.Dataset` の `ReadAsArray` メソッドと同じですが、Float型のNoDataをnp.nanに変換しています。
```python
>>> ary = dst.array()
>>> type(ary)
numpy.ndarray
```

### 1-3. MultiBandRaster（RGB）から Matplotlib で使用するような配列形状のデータを取得
```python
>>> raster = dst.array()
>>> raster.shape
(4, 6000, 6000)
>>> img = dst.array_of_image()
>>> img.shape
(6000, 6000, 4)
```

### 1-4. 新たな配列を書き込んだ、新しい Dataset を作成
```python
>>> import scipy.ndimage
>>> ary = dst.array()
>>> ary = scipy.ndimage.gaussian_filter(ary, sigma=5)
>>> new_dst = dst.write_ary_to_mem(ary)
```

もしもProjectionやGeoTransformは変わらずに、バンド数やデータ型が変わる場合は以下のようにします。
```python
>>> ...
>>> img_ary = ...
>>> new_dst = dst.write_ary_to_mem(
...     ary=ary, 
...     data_type=gdal.GDT_Byte, 
...     out_nodata=255,
...     raster_count=3
... )
```

### 1-5. RasterData の保存
Datasetを保存する場合は以下のようにします。
```python
>>> new_file_path = r'.\new_raster.tif'
>>> dst.save_dst(new_file_path)
```

<br>

## **2. Examples of Projection:**
### 2-1. RasterData の範囲を取得する
```python
>>> import shapely
>>> bounds = dst.bounds
>>> pint(bounds)
Bounds(x_min=-59000.0, y_min=19500.0, x_max=-56500.0, y_max=22000.0)
>>> poly = shapely.box(*bounds)
```

### 2-2. 投影変換した後の範囲を取得する
```python
>>> bounds = dst.reprojected_bounds(4326)
>>> print(bounds)
Bounds(x_min=132.86736372101055, y_min=33.17423487525056, x_max=132.8940138289931, y_max=33.19691043402459)
```

### 2-4. RasterData の投影変換
これは`gdal.Warp`関数と同じですが、投影法だけでなく、投影後の範囲と行列のサイズも指定しています。
```python
>>> dst = gdal.Open(r'.\sample\raster.tif') # EPSG: 4326
>>> out_crs = pyproj.CRS('EPSG:6678')
>>> new_dst = dst.reprojected_dataset(out_crs)
```

### 2-5. RasterData の投影変換（UTM座標系の推定）
```python
>>> new_dst = dst.estimate_utm_and_reprojected_dataset('JGD2011')
```

### 2-6. Resampling（メートル単位で分解能を指定）
EPSG4326のデータセットであっても、何も指定しなければUTMを推定し、引数はメートル単位だと解釈してリサンプリングを行います。
```python
>>> new_dst = dst.resample_with_resol_spec(5, 5)

```

### 2-7. Resampling（分解能を指定）
EPSG:4326のRasterDataをDegree単位でリサンプリングする場合は以下のようにします。
```python
>>> new_dst = dst.resample_with_resol_spec(0.0001, 0.0001, forced_metre_system=False)
``` 

### 2-8. Resampling（セル数を指定）
セル数でリサンプリングする場合は以下のようにします。これはもとのデータセットの範囲を確認してから行う様にしてください。
```python
>>> new_dst = dst.resample_with_cells_spec(100, 100)
``` 

### 2-9. Metre を Degree に変換する
メートル単位の距離を座標系と位置を使用して、度に変換する。
```python
>>> from gdal_drawer.gdal_utils import GdalUtils
>>> gutils = GdalUtils()
>>> metre = 1000
>>> x = -167354.7591972514 
>>> y = -553344.5836010452
>>> in_wkt_crs = pyproj.CRS.from_epsg(6678).to_wkt()
>>> degree = gutils.degree_from_metre(metre, x, y, in_wkt_crs)
0.0109510791
```

### 2-10. Degree を Metre に変換する
度単位の距離を座標系と位置を使用して、メートルに変換する。
```python
>>> gutils = GdalUtils()
>>> degree = 0.0001
>>> lon = 139.00
>>> lat = 35.00
>>> metre = gutils.metre_from_degree(degree, lon, lat)
9.1289
```

### 2-11. セルサイズをメートル単位で取得する
```python
>>> cell_length = dst.cell_size_in_metre(digit=4)
>>> cell_length
CellSize(x_size=0.5, y_size=-0.5)
```

### 2-12. セルサイズを度単位で取得する
```python
>>> cell_length = dst.cell_size_in_degree(digit=9)
>>> cell_length
CellSize(x_size=5.3402e-06, y_size=4.546e-06)
```

<br>

## **3. Examples of Clip and Mask:**
### 3-1. NoDataを埋める
no_dataを埋める場合は以下のようにします。
```python
>>> new_dst = dst.fill_nodata(max_search_distance=10, smoothing=10)
```

### 3-2. RasterDataのクリップ（Geometryでクリップ）
```python
>>> polygon = 'POLYGON((0 0, 0 100, 100 100, 100 0, 0 0))'
>>> new_dst = dst.clip_with_polygon(polygon)
``` 

### 3-3. RasterDataのクリップ（GeometryのBoundingBoxでクリップ）
```python
>>> new_dst = dst.clip_with_bounds(polygon)
```

### 3-4. RasterDataのクリップ（Geometryを含む最小の四角形でクリップ）
```python
>>> new_dst = dst.clip_with_envelope(polygon)
```

### 3-5. RasterDataのマスク処理
これは`np.ndarray`を返すメソッドです。
```python
>>> polygon = 'POLYGON((0 0, 0 100, 100 100, 100 0, 0 0))'
>>> in_wkt_crs = pyproj.CRS('EPSG:4326')
>>> masked_ary = dst.get_masked_array(
...         polygon, # WKT形式のPolygon、またはShapely.geometry
...         in_wkt_crs, # Polygonの投影法
...         masked_value=np.nan, # マスクする値
...         bands=[1, 2, 3], # マスクするバンド
...         all_touched=True, # 交差したセル全てをマスクするか。
...         inverse=False # マスクの反転
...     )
```

<br>

## **4. Examples of processing DTM(DEM):**
### 4-1. 陰影起伏図の作成
```python
>>> new_dst = dst.hillshade(azimuth=315, altitude=90, z_factor=2)
``` 

### 4-2. 傾斜図の作成（gdal.DEMProcessing）
```python
>>> new_dst = dst.slope()
```

### 4-3. 距離を指定して傾斜図を作成
`gdal.DEMProcessing`では隣接セルを使用して勾配を計算しますが、分解能の高いDTMを使用する場合は隣接セルではなく、離れた場所との勾配を計算した方がいい場合もあります。
```python
>>> slope_dst = dst.slope_with_distance_spec(distance=10)
```
DatasetがDegreeでもMetreで指定したい場合は'distance'をメートルで指定し、xyそれぞれのセルサイズを指定する事で可能になります。
```python
>>> cell_size = dst.cell_size_in_metre()
>>> slope_dst = dst.slope_with_distance_spec(
...         distance=10, x_resolution=cell_size.x_size, 
...         y_resolution=cell_size.y_size
...         )
```

### 4-4. セル数を指定して傾斜図を作成
```python
>>> slope_dst = dst.slope_with_cells_spec(x_cells=20, y_cells=20)
```
DatasetがDegreeでもMetreで指定したい場合は'distance'をメートルで指定し、xyそれぞれのセルサイズを指定する事で可能になります。
```python
>>> cell_size = dst.cell_size_in_metre()
>>> slope_dst = dst.slope_with_cells_spec(
...         x_cells=20, y_cells=20, 
...         x_resolution=cell_size.x_size, y_resolution=cell_size.y_size
...         )
```


### 4-5. 方位図の作成
```python
>>> new_dst = dst.aspect()
```

### 4-6. TRIの作成
```python
>>> new_dst = dst.TRI()
```

### 4-7. TPIの作成
```python
>>> # 15mの距離で逆ガウシアンカーネルを作成
>>> kernel = dst.inverse_gaussian_kernel_from_distance(15)
>>> new_dst = dst.TPI(kernel=kernel, outlier_treatment=2.5)
```

### 4-8. TPIの作成（RGBAのラスターとして保存）
```python
>>> # 10mのドーナツカーネルを作成
>>> kernel = dst.doughnut_kernel(10)
>>> tpi_ary = dst.TPI(kernel=kernel, return_ary=True)
>>> # カスタムカラーマップを作成
>>> custom_cmap = CustomCmap()
>>> cmap = custom_cmap.color_list_to_linear_cmap([
...         [0.0, 0.0, 0.8039, 0.6],
...         [0.0039, 0.8353, 1.0, 0.5],
...         [0.5176, 0.0039, 1.0, 0.1],
...         [1.0, 1.0, 1.0, 0.0],
...         [0.9725, 0.3882, 0.0, 0.25],
...         [1.0, 0.1882, 0.0039, 0.5],
...         [1.0, 0.0, 0.0, 0.6]
...      ])
>>> img = cmap.values_to_img(tpi_ary, 'inta')
>>> raster_ary = np.array([img[:, :, i] for i in range(img.shape[2])])
>>> new_dst = dst.write_ary_to_mem(
...         raster_ary, 
...         data_type=gdal.GDT_Int16, 
...         nodata=256, 
...         raster_count=4
...     )
>>> new_dst.save_dst(new_file_path)
```

<br>

## **5. Examples of processing array:**
### 5-1. 正規化した配列を取得
```python
>>> # datasetの配列を正規化
>>> ary = dst.normalized_array()
>>> # 配列を正規化
>>> ary = dst.normalized_array(raster_ary=ary)
```

### 5-2. 標準偏差で外れ値を処理した配列を取得
```python
>>> # datasetの配列にある外れ値を処理
>>> ary = dst.outlier_treatment_array_by_std(2.5)
>>> # 配列にある外れ値を処理
>>> ary = dst.outlier_treatment_array_by_std(2.5, raster_ary=ary)
```

### 5-3. 四分位範囲で外れ値を処理した配列を取得
```python
>>> # datasetの配列にある外れ値を処理
>>> ary = dst.outlier_treatment_array_by_iqr(2.5)
>>> # 配列にある外れ値を処理
>>> ary = dst.outlier_treatment_array_by_iqr(2.5, raster_ary=ary)
```

<br>

## **6. Examples of processing Kernel:**
### 6-1. 平均化カーネルの作成
RasterDataから情報を読み込みメートル単位で距離を指定した平均化カーネルを作成します。
```python
>>> kernel = dst.mean_kernel_from_distance(3)
```

メートル単位で指定したくない場合は以下のようにします。
```python
>> kernel = dst.mean_kernel_from_distance(0.005, metre=False)
```

セル数で平均化カーネルを作成する場合は以下のようにします。
```python
>>> from gdal_drawer.kernels import kernels
>>> kernel = kernels.mean_kernel(x_size=3, y_size=3)
```
### 6-2. ドーナツカーネルの作成
RasterDataから情報を読み込みメートル単位で距離を指定したドーナツカーネルを作成します。
ドーナツカーネルはTPIの計算に使用する為に作成したカーネルです。
```python
>>> kernel = dst.doughnut_kernel_from_distance(5)
```

セル数でドーナツカーネルを作成する場合は以下のようにします。
```python
>>> kernel = kernels.doughnut_kernel(x_size=5, y_size=5)
```

### 6-3. ガウシアンカーネルの作成
```python
>>> kernel = dst.gaussian_kernel_from_distance(5)
```
セル数でガウシアンカーネルを作成する場合は以下のようにします。
```python
>>> kernel = kernels.gaussian_kernel(sigma=5)
```

### 6-4. 逆ガウシアンカーネルの作成
```python
>>> kernel = dst.inverse_gaussian_kernel_from_distance(5)
```
セル数で逆ガウシアンカーネルを作成する場合は以下のようにします。
```python
>>> kernel = kernels.inverse_gaussian_kernel(sigma=5)
```

### 6-5. 畳み込み処理
```python
>>> import scipy.ndimage
>>> ary = dst.array()
>>> kernel = dst.inverse_gaussian_kernel_from_distance(5)
>>> new_ary = scipy.ndimage.convolve(ary, kernel, mode='constant', cval=0.0)
```


<br>

## **7. Examples of visualization:**
### 7-1. ラスターの可視化
```python
>>> fig, ax = plt.subplots()
>>> dst.plot_raster(fig, ax)
>>> plt.show()
```

### 7-2. ラスターの可視化（Nodataを強調して可視化）
```python
>>> fig, ax = plt.subplots()
>>> dst.plot_raster(fig, ax, nodata=True)
>>> plt.show()
```

<br>

## **8. Examples of getting cell coordinates:**
### 8-1. 各セルの座標を取得する
各セルの中心や四隅の座標を取得する場合は以下のようにします。これは`Coordinates`データクラスを返し、`X`と`Y`の2次元配列を持っています。
```python
>>> center_ary = dst.cells_center_coordinates()
>>> center_ary
Coordinates(X=np.array([[0.5, 1.5, 2.5, ..., 97.5, 98.5, 99.5],
                         [0.5, 1.5, 2.5, ..., 97.5, 98.5, 99.5],
                         ...,
                         [0.5, 1.5, 2.5, ..., 97.5, 98.5, 99.5],
                         [0.5, 1.5, 2.5, ..., 97.5, 98.5, 99.5]]),
            Y=np.array([[0.5, 0.5, 0.5, ..., 0.5, 0.5, 0.5],
                         [1.5, 1.5, 1.5, ..., 1.5, 1.5, 1.5],
                         ...,
                         [97.5, 97.5, 97.5, ..., 97.5, 97.5, 97.5],
                         [98.5, 98.5, 98.5, ..., 98.5, 98.5, 98.5]]))
>>> upper_left_ary = dst.cells_upper_left_coordinates()
>>> upper_right_ary = dst.cells_upper_right_coordinates()
>>> lower_left_ary = dst.cells_lower_left_coordinates()
>>> lower_right_ary = dst.cells_lower_right_coordinates()
```

### 8-2. 各セルの座標を取得する（GeoDataFrame）
各セルの指定した場所の座標とセルの値を取得し、GeoDataFrameを作成します。
```python
>>> # セル中心座標を取得して、GeoDataFrameを作成する
>>> center_gdf = dst.to_geodataframe_xy(position='center')
>>> # セル左上の座標を取得して、GeoDataFrameを作成する
>>> upper_left_gdf = dst.to_geodataframe_xy(position='upper_left')
>>> # セル右上の座標を取得して、GeoDataFrameを作成する
>>> upper_right_gdf = dst.to_geodataframe_xy(position='upper_right')
>>> # セル左下の座標を取得して、GeoDataFrameを作成する
>>> lower_left_gdf = dst.to_geodataframe_xy(position='lower_left')
>>> # セル右下の座標を取得して、GeoDataFrameを作成する
>>> lower_right_gdf = dst.to_geodataframe_xy(position='lower_right')
```

### 8-3. 各セルの座標を取得する（DataFrame）
pandas.DataFrameを返すメソッドもあります。DataFrameの場合は'geometry'が Wkt形式で返されます。
```python
>>> df = dst.to_dataframe_xy(position='center')
```