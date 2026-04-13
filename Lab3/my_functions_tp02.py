import tools
import matplotlib.pyplot as plt
import numpy as np
def get_gaussian_filtre(dimension=3,sigma=0.5):
    """return the result after apply a gaussian filtre with a given dimension """
    kernel = tools.gaussian_mask(size=dimension,sigma=sigma)
    return kernel
def plot_img(img,title):
    fig,ax = plt.subplots()
    ax.imshow(img)
    ax.set_title(title)

def filter_analysis(img,kernel):
    """apply filtre on imageand compare between original and filtred image"""
    filtred_im =apply_filter_to_single_channel(img,kernel)
    fig,ax = plt.subplots(1,2)
#     ploting original img
    ax[0].imshow(img)
    ax[0].set_title("original")
#     ploting filtred img
    ax[1].imshow(filtred_im)
    ax[1].set_title('filtred image')
def apply_filter_to_single_channel(img,kernel):
    dimK = kernel.shape
    return tools.Conv2D(tools.add_padding(img,((dimK[0]-1)//2,(dimK[1]-1)//2)),kernel)
def apply_filter_to_colored_img(img,kernel):
#     dstack build ndarray on the third axis
    return np.dstack([apply_filter_to_single_channel(img[:,:,z],kernel) for z in range(3)])
def filter_analysis_colored(img,kernel):
    """apply filtre on imageand compare between original and filtred image"""
   
    filtred_im = apply_filter_to_colored_img(img,kernel)
    fig,ax = plt.subplots(1,2)
#     ploting original img
    ax[0].imshow(img)
    ax[0].set_title("original")
#     ploting filtred img
    ax[1].imshow(filtred_im)
    ax[1].set_title('filtred image')
def get_Gx(img):
#     get img conv Sx filter
    Sx = np.array([[-1,0,1],[-2,0,2],[-1,0,-1]])
    return apply_filter_to_single_channel(img,Sx)
def get_Gy(img):
#     get img conv Sy filter
    Sy = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    return apply_filter_to_single_channel(img,Sy)
# filtre laplace
def get_L(img):
    L = np.array([[0,1,0],[-1,4,-1],[0,1,0]])
    return apply_filter_to_single_channel(img,L)
def module_grad(img):
    Gx = get_Gx(img)
    Gy = get_Gy(img)
    result=np.sqrt(Gx**2+Gy**2)
    return result
def direct_grad(img):
    Gx = get_Gx(img)
    Gy = get_Gy(img)
    result=np.arctan(Gy/Gx)
    return result