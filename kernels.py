from typing import NamedTuple

from matplotlib import pyplot as plt
import numpy as np


class KernelSize(NamedTuple):
    x: int
    y: int


class Kernels(object):
    @staticmethod
    def _adjust_size(arg_index: int, arg_name: str):
        """カーネルサイズを奇数に調整するデコレーター"""        
        def decorater(func):
            def wrapper(self, *args, **kwargs):
                size = None
                in_args = True
                if arg_index < len(args):
                    size = args[arg_index]
                elif arg_name in kwargs:
                    size = kwargs[arg_name]
                    in_args = False
                else:
                    return func(self, *args, **kwargs)
                _, remainder = divmod(size, 2)
                if 0 == remainder:
                    size += 1
                if in_args:
                    args = list(args)
                    args[arg_index] = size
                else:
                    kwargs[arg_name] = size
                return func(self, *args, **kwargs)
            return wrapper
        return decorater

    def distance_to_kernel_size(self,
        distance: float, 
        x_cell_size: float, 
        y_cell_size: float=None
    ) -> KernelSize:
        """
        辺の距離とセルの長さからカーネルのサイズを計算する。
        Args:
            distance(int): 辺の長さ
            x_cell_size(int): x方向のセルサイズ
            y_cell_size(int): y方向のセルサイズ
        Returns:
            cells(int): カーネルのサイズ。これは必ず奇数になる。
        Examples:
            >>> distance_to_kernel_size(3, 1)
            3
            >>> distance_to_kernel_size(3, 2)
            5
        """
        if y_cell_size is None:
            y_cell_size = x_cell_size
        x_divd, x_remainder = divmod(distance, abs(x_cell_size))
        if 0 < x_remainder:
            x_divd += 1
        y_divd, y_remainder = divmod(distance, abs(y_cell_size))
        if 0 < y_remainder:
            y_divd += 1
        return KernelSize(int(x_divd), int(y_divd))    

    @_adjust_size(0, "x_size")
    @_adjust_size(1, "y_size")
    def mean_kernel(self, x_size: int, y_size: int=None) -> np.ndarray:
        """
        平均化カーネルを作成する。
        Args:
            x_size(int): X方向のカーネルのサイズ
            y_size(int): Y方向のカーネルのサイズ
        Returns:
            kernel(np.ndarray): 平均化カーネル
        Examples:
            >>> mean_kernel(3)
            array([[0.11111111, 0.11111111, 0.11111111],
                   [0.11111111, 0.11111111, 0.11111111],
                   [0.11111111, 0.11111111, 0.11111111]])
        """
        if y_size is None:
            y_size = x_size
        return np.ones((y_size, x_size)) / (x_size * y_size)
    
    @_adjust_size(0, "x_size")
    @_adjust_size(1, "y_size")
    def doughnut_kernel(self, x_size: int, y_size: int=None) -> np.ndarray:
        """
        畳み込みに使用するドーナツ型のカーネルを生成する。ドーナツ型のカーネルは外周にのみ値が入力されており、足し合わせると1になる。
        Args:
            distance(int): カーネルのサイズ
        Returns:
            np.ndarray
        Examples:
            >>> doughnut_kernel(5)
            array([[0.0625, 0.0625, 0.0625, 0.0625, 0.0625],
                   [0.0625, 0.    , 0.    , 0.    , 0.0625],
                   [0.0625, 0.    , 0.    , 0.    , 0.0625],
                   [0.0625, 0.    , 0.    , 0.    , 0.0625],
                   [0.0625, 0.0625, 0.0625, 0.0625, 0.0625]])
        """
        if y_size is None:
            y_size = x_size
        shape = (x_size, y_size)
        outer_cells = (x_size - 1) * 2 + (y_size - 1) * 2
        input_val = 1 / outer_cells
        kernel = np.zeros(shape)
        kernel[:1] = input_val
        kernel[-1:] = input_val
        kernel[:, :1] = input_val
        kernel[:, -1:] = input_val
        return kernel
    
    def gaussian_kernel(self, sigma: float, coef: float=None) -> np.ndarray:
        """
        ガウシアンフィルターの為のカーネルを作成する。
        Args:
            sigma(float): ガウス分布の標準偏差
            coef(float): カーネルの値を強調する係数
        Returns:
            kernel(np.ndarray): ガウシアンフィルターのカーネル
        """
        size = int(2 * np.ceil(3 * sigma) + 1)  
        ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
        kernel = kernel / np.sum(kernel)
        if coef is not None:
            add = (coef - 1) / (size * size)
            kernel = kernel * coef - add
        return kernel

    def inverse_gaussian_kernel(self, sigma: float, coef: float=None) -> np.ndarray:
        """
        逆ガウシアンフィルターの為のカーネルを作成する。
        Args:
            sigma(float): ガウス分布の標準偏差
            coef(float): カーネルの値を強調する係数
        Returns:
            kernel(np.ndarray): 逆ガウシアンフィルターのカーネル
        """
        kernel = self.gaussian_kernel(sigma, coef) * -1
        rows, cols = kernel.shape
        kernel += 2 / (rows * cols)
        return kernel

    @_adjust_size(0, "x_size")
    @_adjust_size(1, "y_size")
    def gaussian_kernel_from_size(self, 
        x_size: int, 
        y_size: int=None, 
        coef: float=None
    ) -> np.ndarray:
        """ 
        ガウシアンフィルターをカーネルサイズから作成
        Args:
            x_size(int): X方向のカーネルのサイズ
            y_size(int): Y方向のカーネルのサイズ
            coef(float): カーネルの値を強調する係数
        Returns:
            kernel(np.ndarray): ガウシアンフィルターのカーネル
        """
        if y_size is None:
            y_size = x_size
        sigma_x = (x_size - 1) / (2 * 3)
        sigma_y = (y_size - 1) / (2 * 3)
        ax_x = np.linspace(-(x_size - 1) / 2., (x_size - 1) / 2., x_size)
        ax_y = np.linspace(-(y_size - 1) / 2., (y_size - 1) / 2., y_size)
        xx, yy = np.meshgrid(ax_x, ax_y)
        kernel = np.exp(
            -0.5 * (np.square(xx) / np.square(sigma_x) 
            + np.square(yy) / np.square(sigma_y))
        )
        kernel = kernel / np.sum(kernel)
        if coef is not None:
            add = (coef - 1) / (x_size * y_size)
            kernel = kernel * coef - add
        return kernel
    
    @_adjust_size(0, "x_size")
    @_adjust_size(1, "y_size")
    def inverse_gaussian_kernel_from_size(self, 
        x_size: int,
        y_size: int=None,
        coef: float=None
    ) -> np.ndarray:
        """ 
        逆ガウシアンフィルターをカーネルサイズから作成
        Args:
            x_size(int): X方向のカーネルのサイズ
            y_size(int): Y方向のカーネルのサイズ
            coef(float): カーネルの値を強調する係数
        Returns:
            kernel(np.ndarray): 逆ガウシアンフィルターのカーネル
        """
        if y_size is None:
            y_size = x_size
        kernel = self.gaussian_kernel_from_size(x_size, y_size, coef) * -1
        rows, cols = kernel.shape
        kernel += 2 / (rows * cols)
        return kernel

    def plot_kernel_3d(self, 
        kernel: np.ndarray, 
        cmap_name='bwr',
        **kwargs
    ) -> None:
        """
        カーネルを3Dでプロットする。
        Args:
            kernel(np.ndarray): カーネル
            cmap_name(str): カラーマップの名前
            kwargs:
                - unit_length(float): セルの大きさ（距離）
                - is_marker(bool): markerにするかどうか。デフォルトはFalse
        """
        # X, Y軸のインデックスを生成
        _X = self.__generate_index(kernel[0, :])
        _Y = self.__generate_index(kernel[:, 0])
        X, Y = np.meshgrid(_X, _Y)
        # カラーマップを生成
        norm = plt.Normalize(kernel.min(), kernel.max()) 
        cmap = plt.get_cmap(cmap_name)
        colors = cmap(norm(kernel))
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        if kwargs.get('is_marker', False):
            # 散布図でプロット（douguhnutカーネルの場合は、色がおかしくなるので）
            ax.plot_wireframe(X, Y, kernel, lw=1, ec='gray', rstride=1, cstride=1)
            ax.scatter(X, Y, kernel, c=colors.reshape(-1, 4), marker='o', s=100)
        else:
            ax.scatter(X, Y, kernel, c=colors.reshape(-1, 4), marker='o', s=10)
            ax.plot_trisurf(
                X.flatten(), Y.flatten(), kernel.flatten(), norm=norm,
                lw=0.5, alpha=0.7, cmap=cmap_name
            )
        # ターゲットセルの位置に垂直線を追加
        ax.plot(
            [0, 0], [0, 0], 
            [np.min(kernel), np.max(kernel)], 
            color='black', lw=10,
            label='Target cell'
        )
        # ColorBarを追加
        mappable = plt.cm.ScalarMappable(cmap=cmap_name, norm=norm) 
        mappable.set_array(kernel) 
        fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=5)
        if 'unit_length' in kwargs:
            row_length = kernel.shape[0] * kwargs['unit_length']
            col_length = kernel.shape[1] * kwargs['unit_length']
            title = (f"Kernel shape: {kernel.shape}\n"
                    f"Distance: ({row_length}, {col_length})")
        else:
            title = f"Kernel shape: {kernel.shape}"

        ax.set_title(title, fontsize=15, fontweight='bold')
        ax.set_xlabel('X', fontsize=15)
        ax.set_ylabel('Y', fontsize=15)
        ax.set_zlabel('Weight', fontsize=15)
        ax.legend(fontsize=10, loc='upper right')
        plt.show()
        
    def __generate_index(self, ary1d: list):
        divid, over = divmod(len(ary1d), 2)
        left_idx = np.arange(0, -divid, -1) - 1
        left_idx.sort()
        if over == 1:        
            right_idx = np.arange(0, divid, 1) + 1
        else:
            right_idx = np.arange(0, divid - 1, 1) + 1
        return left_idx.tolist() + [0] + right_idx.tolist()



kernels = Kernels()