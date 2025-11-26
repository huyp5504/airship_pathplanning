


# import numpy as np
# import netCDF4 as nc
# import matplotlib.pyplot as plt
# class WindFieldAssimilation:
#     def __init__(self,f_path, window_size=32, process_noise=0.1, measurement_noise=0.2, decay_factor=5):
#         self.window_size = window_size

#         self.decay_factor = decay_factor  # Controls Gaussian distance weight decay

#         # Initialize wind field (u, v) and uncertainty
#         self.wind_u = np.zeros((window_size, window_size))  # u component
#         self.wind_v = np.zeros((window_size, window_size))  # v component
#         self.uncertainty = np.ones((window_size, window_size))  # Initial uncertainty
#         self.f_path =f_path 
#         self.dataset=nc.Dataset(f_path)

#     def load_forecast(self,position=[32,32]):
#         """
#         Load wind field forecast data from a NetCDF file.
#         Assumes variables 'u' and 'v' are present in the file.
#         """
#         [x,y]=position
#         self.wind_u=self.extract_submatrix(self.dataset.variables['u'][6][0][0]/10, [x,y])
#         self.wind_v=self.extract_submatrix(self.dataset.variables['v'][6][0][0]/10, [x,y])
#     def extract_submatrix(self,field, center):

#         # get the given center position
#         i, j = center
#         # print(center)
#         # print(field)
#         # half-size of the 32x32 matrix
#         half_size = self.window_size/2
        
#         # initialize a 32x32 zero matrix
#         submatrix = np.zeros((self.window_size, self.window_size))
        

#         for i in range(self.window_size):
#             for j in range(self.window_size):
#                 # compute offset from the given center position
#                 # print(i + center[0] - half_size, j + center[1] - half_size)
#                 try:
#                     submatrix[i, j] = field[int(i + center[0] - half_size), int(j + center[1] - half_size)]
#                 except:
#                     submatrix[i, j] = 0
#         # print(submatrix)
#         print(submatrix.shape)
#         return submatrix
    
    
#     def fuse_wind_field(self,postion,measured_u,measured_v, sigma=5):
#         """
#         Fuse accurate wind measurements at the airship center with the forecast wind field.

#         Parameters:
#         - wind_f: 2D numpy array (32x32), forecast wind field.
#         - W_m: float, measured wind speed at the airship position.
#         - center_x, center_y: int, center indices in array (16, 16).
#         - sigma: float, controls the influence range.

#         Returns:
#         - wind_u: 2D numpy array, updated wind field.
#         - uncertainty: 2D numpy array, per-cell uncertainty.
#         """
#         center_x, center_y = [16,16]
#         # 初始化数组
        
        
#         # difference between measured and forecast values at center
#         E_u_center = self.wind_u[center_x, center_y]
#         E_v_center = self.wind_v[center_x, center_y]
#         delta_u = E_u_center - measured_u
#         delta_v = E_v_center - measured_v
#         error = np.sqrt(delta_u**2+delta_v**2)
#         # base uncertainty
#         U_base = 0.2  # base uncertainty, adjust as needed
#         U_delta = 1.0  # uncertainty scale factor based on delta, adjust as needed

#         # create coordinate grid
#         x = np.arange(self.window_size)
#         y = np.arange(self.window_size)
#         X, Y = np.meshgrid(x, y, indexing='ij')

#     # compute distance from center
#         D = np.sqrt((X - center_x)**2 + (Y - center_y)**2)*(1/error)*0.2

#         #
#     # compute influence function
#         I = np.exp(-D**2 / (2 * sigma**2))

#         # update wind field
#         self.wind_u = self.wind_u - delta_u * I
#         self.wind_v = self.wind_v - delta_v * I

#         # compute uncertainty
#         self.uncertainty = U_base + U_delta * np.abs(error) * (1 - I)


        

#     # In this example, we assume extracting the current timestep data
#     def update(self, position, measured_u, measured_v):
#         """
#         Update wind field based on measurements.
#         """
#         self.load_forecast(position)
        
#         # Create figure with 3 subplots
#         fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 7))
        
#         # Plot initial wind field
#         u = self.wind_u[::2, ::2]
#         v = self.wind_v[::2, ::2]
#         x, y = np.meshgrid(np.arange(0, self.window_size, 2), np.arange(0, self.window_size, 2))
#         speed = np.ones(u.shape)
#         ax1.quiver(x, y, u*3, v*3, speed, cmap='jet', scale=50)
#         ax1.set_title("Initial Wind Field")
#         ax1.set_aspect('equal')
        
#         # Fuse wind field
#         self.fuse_wind_field(position, measured_u, measured_v)
        
#         # Plot updated wind field
#         u = self.wind_u[::2, ::2]
#         v = self.wind_v[::2, ::2]
#         speed = np.sqrt(u**2 + v**2)
#         ax2.quiver(x, y, u*3, v*3, 1-self.uncertainty[::2,::2], cmap='jet', scale=50)
#         ax2.set_title("Updated Wind Field")
#         ax2.set_aspect('equal')
        
#         # Plot uncertainty field
#         im = ax3.imshow(self.uncertainty, cmap='hot', extent=[0, self.window_size, 0, self.window_size])
#         ax3.set_title("Uncertainty Field After Assimilation")
#         ax3.set_aspect('equal')
        
#         # Add colorbar with adjusted padding
#         cbar = fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
        
#         # Adjust layout and show combined plot
#         plt.tight_layout()
#         plt.show()


# if __name__ == "__main__":
#     assimilation = WindFieldAssimilation(window_size=32,f_path='adaptor.mars.internal-1708914676.224747-22914-14-6a4c2165-bf79-4602-861d-5d1a0cec3f51.nc')
#     # assimilation.update([44,111],0.5,0)
#     assimilation.update([88,125],0.5,0.1)

# # The updated wind field (wind_u) and uncertainty map (uncertainty) are now available
# # They can be used for further processing in reinforcement learning algorithms
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import os

class WindFieldAssimilation:
    def __init__(self, f_path, window_size=32, process_noise=0.1, measurement_noise=0.2, decay_factor=5):
        self.window_size = window_size
        self.decay_factor = decay_factor  # Controls Gaussian distance weight decay

        # Initialize wind field (u, v) and uncertainty
        self.wind_u = np.zeros((window_size, window_size))  # u component
        self.wind_v = np.zeros((window_size, window_size))  # v component
        self.uncertainty = np.ones((window_size, window_size))  # Initial uncertainty
        self.f_path = f_path
        self.dataset = nc.Dataset(f_path)

    def load_forecast(self, position=[32, 32]):
        """
        Load wind field forecast data from a NetCDF file.
        Assumes variables 'u' and 'v' are present in the file.
        """
        [x, y] = position
        self.wind_u = self.extract_submatrix(self.dataset.variables['u'][6][0][0] / 10, [x, y])
        self.wind_v = self.extract_submatrix(self.dataset.variables['v'][6][0][0] / 10, [x, y])

    def extract_submatrix(self, field, center):
        # get the given center position
        i, j = center
        # half-size of the 32x32 matrix
        half_size = self.window_size / 2

        # initialize a 32x32 zero matrix
        submatrix = np.zeros((self.window_size, self.window_size))

        for i in range(self.window_size):
            for j in range(self.window_size):
                # compute offset from the given center position
                try:
                    submatrix[i, j] = field[int(i + center[0] - half_size), int(j + center[1] - half_size)]
                except:
                    submatrix[i, j] = 0
        return submatrix

    def fuse_wind_field(self, postion, measured_u, measured_v, sigma=5):
        """
        Fuse accurate wind measurements at the airship center with the forecast wind field.
        """
        center_x, center_y = [16, 16]
        # difference between measured and forecast values at center
        E_u_center = self.wind_u[center_x, center_y]
        E_v_center = self.wind_v[center_x, center_y]
        delta_u = E_u_center - measured_u
        delta_v = E_v_center - measured_v
        error = np.sqrt(delta_u**2 + delta_v**2)
        # base uncertainty
        U_base = 0.2  # base uncertainty, adjust as needed
        U_delta = 1.0  # uncertainty scale factor based on delta, adjust as needed

        # create coordinate grid
        x = np.arange(self.window_size)
        y = np.arange(self.window_size)
        X, Y = np.meshgrid(x, y, indexing='ij')

        # compute distance from center
        D = np.sqrt((X - center_x)**2 + (Y - center_y)**2) * (1 / error) * 0.2

        # compute influence function
        I = np.exp(-D**2 / (2 * sigma**2))

        # update wind field
        self.wind_u = self.wind_u - delta_u * I
        self.wind_v = self.wind_v - delta_v * I

        # compute uncertainty
        self.uncertainty = U_base + U_delta * np.abs(error) * (1 - I)

    def update(self, position, measured_u, measured_v, save_dir="output_images"):
        """
        Update wind field based on measurements and save images to the specified directory.
        """
        # create output directory if it does not exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.load_forecast(position)

        # Plot initial wind field
        plt.figure(figsize=(7, 7), dpi=500)
        u = self.wind_u[::2, ::2]
        v = self.wind_v[::2, ::2]
        x, y = np.meshgrid(np.arange(0, self.window_size, 2), np.arange(0, self.window_size, 2))
        speed = np.ones(u.shape)
        plt.quiver(x, y, u * 3, v * 3, speed, cmap='jet', scale=50)
        plt.gca().set_aspect('equal')
        
        # add caption below the image
        plt.text(0.5, -0.1, "(a) Initial Wind", transform=plt.gca().transAxes, 
                 fontsize=12, ha='center', va='center')
        
        plt.savefig(os.path.join(save_dir, "initial_wind_field.png"), bbox_inches='tight')
        plt.close()

        # Fuse wind field
        self.fuse_wind_field(position, measured_u, measured_v)

        # Plot updated wind field
        plt.figure(figsize=(7, 7), dpi=500)
        u = self.wind_u[::2, ::2]
        v = self.wind_v[::2, ::2]
        speed = np.sqrt(u**2 + v**2)
        plt.quiver(x, y, u * 3, v * 3, 1 - self.uncertainty[::2, ::2], cmap='jet', scale=50)
        plt.gca().set_aspect('equal')
        
        # add caption below the image
        plt.text(0.5, -0.1, "(b) Updated Wind", transform=plt.gca().transAxes, 
                 fontsize=12, ha='center', va='center')
        
        plt.savefig(os.path.join(save_dir, "updated_wind_field.png"), bbox_inches='tight')
        plt.close()

        # Plot uncertainty field
        plt.figure(figsize=(7, 7), dpi=500)
        plt.imshow(self.uncertainty, cmap='hot', extent=[0, self.window_size, 0, self.window_size])
        plt.gca().set_aspect('equal')
        plt.colorbar(fraction=0.046, pad=0.04)
        
        # add caption below the image
        plt.text(0.5, -0.1, "(c) Uncertainty After Assimilation", transform=plt.gca().transAxes, 
                 fontsize=12, ha='center', va='center')
        
        plt.savefig(os.path.join(save_dir, "uncertainty_field.png"), bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    assimilation = WindFieldAssimilation(window_size=32, f_path='adaptor.mars.internal-1708914676.224747-22914-14-6a4c2165-bf79-4602-861d-5d1a0cec3f51.nc')
    assimilation.update([88, 125], 0.5, 0.1, save_dir="output_images")
