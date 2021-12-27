"""Stereo matching."""
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from scipy.ndimage import rotate


class Solution:
    def __init__(self):
        pass

    @staticmethod
    def ssd_distance(left_image: np.ndarray,
                     right_image: np.ndarray,
                     win_size: int,
                     dsp_range: int) -> np.ndarray:
        """Compute the SSDD distances tensor.

        Args:
            left_image: Left image of shape: HxWx3, and type np.double64.
            right_image: Right image of shape: HxWx3, and type np.double64.
            win_size: Window size odd integer.
            dsp_range: Half of the disparity range. The actual range is
            -dsp_range, -dsp_range + 1, ..., 0, 1, ..., dsp_range.

        Returns:
            A tensor of the sum of squared differences for every pixel in a
            window of size win_size X win_size, for the 2*dsp_range + 1
            possible disparity values. The tensor shape should be:
            HxWx(2*dsp_range+1).
        """
        num_of_rows, num_of_cols = left_image.shape[0], left_image.shape[1]
        disparity_values = range(-dsp_range, dsp_range+1)
        ssdd_tensor = np.zeros((num_of_rows,
                                num_of_cols,
                                len(disparity_values)))
        """INSERT YOUR CODE HERE"""

        # now iv leard that we need to use the convolve2d func so we couled solve accordinley
        kernel= np.ones((win_size,win_size))
        #pad_width=()
        new_rigt_image=np.pad(right_image, [dsp_range,dsp_range], mode='constant', constant_values=(0, 0))
        new_rigt_image=new_rigt_image[dsp_range:-dsp_range,:,dsp_range:-dsp_range]
        for dis in disparity_values:
            
            temp_right_img=np.roll (new_rigt_image,dis,axis=1)
            #plt.figure()
        
            #plt.imshow(temp_right_img)
            temp_right_img=temp_right_img[:,dsp_range:-dsp_range,:]
            '''
            if (dis% 3==0) :
                plt.figure()
        
                plt.imshow(temp_right_img)

            '''
            calc_movment= (left_image-temp_right_img)**2
            for color in range(3):
                #now we can sum over the colors
                
                #temp =(convolve2d(left_image[:,:,color],kernel,mode='same')-convolve2d(temp_right_img[:,:,color],kernel,mode='same'))**2#,boundary='symm'))**2
                temp =convolve2d(calc_movment[:,:,color],kernel,mode='same')
                ssdd_tensor[:,:,dis]+= temp    

        ssdd_tensor -= ssdd_tensor.min()
        ssdd_tensor /= ssdd_tensor.max()
        ssdd_tensor *= 255.0
        return ssdd_tensor

    @staticmethod
    def naive_labeling(ssdd_tensor: np.ndarray) -> np.ndarray:
        """Estimate a naive depth estimation from the SSDD tensor.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for every
            pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.

        Evaluate the labels in a naive approach. Each value in the
        result tensor should contain the disparity matching minimal ssd (sum of
        squared difference).

        Returns:
            Naive labels HxW matrix.
        """
        # you can erase the label_no_smooth initialization.
        label_no_smooth = np.zeros((ssdd_tensor.shape[0], ssdd_tensor.shape[1]))
        

        """INSERT YOUR CODE HERE"""
        for i in range(ssdd_tensor.shape[0]):
            for j in range(ssdd_tensor.shape[1]):
                #min = np.amin(ssdd_tensor[i,j,:])
                #temp1 = np.argmin(ssdd_tensor[i,j,:]) 
                #temp = np.where(ssdd_tensor[i,j,:]== min )
                #temp2= temp[0]
                #if temp2[0] != temp1:
                #    print(f' the value of np where {temp2[0]} is diffrent from arg min {temp1}')
                temp1 = ssdd_tensor[i,j,:]  
                temp=np.argmin(ssdd_tensor[i,j,:])
                label_no_smooth[i,j] = temp
        return label_no_smooth.astype(int)

    @staticmethod
    def dp_grade_slice(c_slice: np.ndarray, p1: float, p2: float) -> np.ndarray:
        """Calculate the scores matrix for slice c_slice.

        Calculate the scores slice which for each column and disparity value
        states the score of the best route. The scores slice is of shape:
        (2*dsp_range + 1)xW.

        Args:
            c_slice: A slice of the ssdd tensor.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.
        Returns:
            Scores slice which for each column and disparity value states the
            score of the best route.
        """
        num_labels, num_of_cols = c_slice.shape[0], c_slice.shape[1]
        l_slice = np.zeros((num_labels, num_of_cols))
        #initialize the DB
        l_slice[:,0] = c_slice[:,0]
        
        """INSERT YOUR CODE HERE"""
        M= np.zeros((num_labels, num_of_cols))
        #initialize the start penlety to be a inf num so we couled find the 
        #path
        M[:,0] = l_slice[:,0]
        
        for col in range(1,num_of_cols):
            for d in range(num_labels):
                M[d,col]=l_slice[d,col-1]
                temp3= l_slice[d,col-1]
                if  d>0 :
                    M[d,col] = min(M[d,col],p1 +l_slice[d-1,col-1] )
                if (d+1)<num_labels:
                        M[d,col] = min(M[d,col],p1 +l_slice[d+1,col-1]) 
                for k in range(2,num_labels):
                    if (d+k) < num_labels:
                       M[d,col] =min(M[d,col],p2 + l_slice[d+k,col-1])  
                    if (d-k) >= 0:
                        M[d,col] =min(M[d,col],p2 + l_slice[d-k,col-1])  
                '''
                for k in range(d+2,num_labels):
                    M[d,col] =min(M[d,col],p2 + l_slice[k,col-1])
                for k in range(d-2,-1,-1):
                    M[d,col] = min(M[d,col],p2 + l_slice[k,col-1])
                '''
                l_slice[d,col]= c_slice[d,col] + M[d,col] - min(l_slice[:,col-1])

                    


        return l_slice

    def dp_labeling(self,
                    ssdd_tensor: np.ndarray,
                    p1: float,
                    p2: float) -> np.ndarray:
        """Estimate a depth map using Dynamic Programming.

        (1) Call dp_grade_slice on each row slice of the ssdd tensor.
        (2) Store each slice in a corresponding l tensor (of shape as ssdd).
        (3) Finally, for each pixel in l (along each row and column), choose
        the best disparity value. That is the disparity value which
        corresponds to the lowest l value in that pixel.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for every
            pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.
        Returns:
            Dynamic Programming depth estimation matrix of shape HxW.
        """
        l = np.zeros_like(ssdd_tensor)
        for row in range(ssdd_tensor.shape[0]):
        
               temp= ssdd_tensor[row,:,:]
               temp2=self.dp_grade_slice(ssdd_tensor[row,:,:].T,p1,p2).T
               l[row,:]= temp2
        """INSERT YOUR CODE HERE"""
        return self.naive_labeling(l)

    
    def dp_labeling_L(self,
                    ssdd_tensor: np.ndarray,
                    p1: float,
                    p2: float) -> np.ndarray:
        """Estimate a depth map using Dynamic Programming.

        (1) Call dp_grade_slice on each row slice of the ssdd tensor.
        (2) Store each slice in a corresponding l tensor (of shape as ssdd).
        (3) Finally, for each pixel in l (along each row and column), choose
        the best disparity value. That is the disparity value which
        corresponds to the lowest l value in that pixel.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for every
            pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.
        Returns:
            Dynamic Programming depth estimation matrix of shape HxWxd.
        """
        l = np.zeros_like(ssdd_tensor)
        for row in range(ssdd_tensor.shape[0]):
        
               temp= ssdd_tensor[row,:,:]
               l[row,:]= self.dp_grade_slice(ssdd_tensor[row,:,:].T,p1,p2).T
        """INSERT YOUR CODE HERE"""
        return (l)


    def extract_slices(self,
                    ssdd_tensor: np.ndarray,
                    direction: int,
                    p1: float,
                    p2: float) -> np.ndarray:
        '''
    Args:
            ssdd_tensor: A tensor of the sum of squared differences for every
            pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            direction: the wanted direction to compute the depth astimation along it
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.
        Returns:
            Dynamic Programming depth estimation matrix of shape HxW.

        '''
        if direction ==1:
            return self.dp_labeling_L(ssdd_tensor,p1,p2)
        elif direction == 3:
            temp = (self.dp_labeling_L(np.rot90(ssdd_tensor),p1,p2))
            temp2 =np.flipud(np.fliplr(np.rot90(temp)))
            return  temp2
        elif direction == 5:
            temp = self.dp_labeling_L(np.fliplr(ssdd_tensor),p1,p2)
            return  np.fliplr(temp)
        elif direction == 7:
            temp = self.dp_labeling_L(np.fliplr(np.rot90(ssdd_tensor)),p1,p2)
            #temp2 = np.fliplr(temp).T 
            temp2 = np.fliplr(np.rot90(temp))
            return  temp2
        elif direction == 2:
            return self.dp_diag_labeling_L(ssdd_tensor,p1,p2)
        elif direction == 4:
            temp = self.dp_diag_labeling_L(np.fliplr(ssdd_tensor),p1,p2)
            return  np.fliplr(temp)
        elif direction == 6:
            temp = self.dp_diag_labeling_L(np.fliplr(np.rot90(ssdd_tensor)),p1,p2)
            temp2 = np.fliplr(np.rot90(temp))
            #return  np.fliplr(np.fliplr(temp).T)
            return  temp2
        elif direction == 8:
            temp = self.dp_diag_labeling_L(np.rot90(ssdd_tensor),p1,p2)
            temp2 = np.flipud(np.fliplr(np.rot90(temp)))
            #return np.rot90(np.fliplr(np.fliplr(temp).T))
            return temp2
        return
         
    def dp_diag_labeling(self,
                    ssdd_tensor: np.ndarray,
                    p1: float,
                    p2: float) -> np.ndarray:
        """Estimate a depth map using Dynamic Programming.

        (1) Call dp_grade_slice on each row slice of the ssdd tensor.
        (2) Store each slice in a corresponding l tensor (of shape as ssdd).
        (3) Finally, for each pixel in l (along each row and column), choose
        the best disparity value. That is the disparity value which
        corresponds to the lowest l value in that pixel.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for every
            pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.
        Returns:
            Dynamic Programming depth estimation matrix of shape HxW.
        """
        l = np.zeros_like(ssdd_tensor)
        x_axis = np.arange(ssdd_tensor.shape[1])
        y_axis = np.arange(ssdd_tensor.shape[0])
        xx,yy= np.meshgrid(x_axis, y_axis, sparse=False, indexing='xy')
        
        for col in range(ssdd_tensor.shape[1]):
        
               temp= ssdd_tensor.diagonal(col)
               temp2= xx.diagonal(col)
               temp3 = yy.diagonal(col)
               location = [temp2,temp3]
               l[location[1],location[0],:]= self.dp_grade_slice(temp,p1,p2).T
        
        for row in range(ssdd_tensor.shape[0]):
        
               temp= ssdd_tensor.diagonal(-row)
               temp2= xx.diagonal(-row)
               temp3 = yy.diagonal(-row)
               location = [temp2,temp3]
               l[location[1],location[0],:]= self.dp_grade_slice(temp,p1,p2).T

               
               
       
        """INSERT YOUR CODE HERE"""
        return self.naive_labeling(l)

    def dp_diag_labeling_L(self,
                    ssdd_tensor: np.ndarray,
                    p1: float,
                    p2: float) -> np.ndarray:
        """Estimate a depth map using Dynamic Programming.

        (1) Call dp_grade_slice on each row slice of the ssdd tensor.
        (2) Store each slice in a corresponding l tensor (of shape as ssdd).
        (3) Finally, for each pixel in l (along each row and column), choose
        the best disparity value. That is the disparity value which
        corresponds to the lowest l value in that pixel.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for every
            pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.
        Returns:
            Dynamic Programming depth estimation matrix of shape HxWxd.
        """
        l = np.zeros_like(ssdd_tensor)
        x_axis = np.arange(ssdd_tensor.shape[1])
        y_axis = np.arange(ssdd_tensor.shape[0])
        xx,yy= np.meshgrid(x_axis, y_axis, sparse=False, indexing='xy')
        
        for col in range(ssdd_tensor.shape[1]):
        
               temp= ssdd_tensor.diagonal(col)
               temp2= xx.diagonal(col)
               temp3 = yy.diagonal(col)
               location = [temp2,temp3]
               l[location[1],location[0],:]= self.dp_grade_slice(temp,p1,p2).T
        
        for row in range(ssdd_tensor.shape[0]):
        
               temp= ssdd_tensor.diagonal(-row)
               temp2= xx.diagonal(-row)
               temp3 = yy.diagonal(-row)
               location = [temp2,temp3]
               l[location[1],location[0],:]= self.dp_grade_slice(temp,p1,p2).T

               
               
       
        """INSERT YOUR CODE HERE"""
        return (l)

    def dp_labeling_per_direction(self,
                                  ssdd_tensor: np.ndarray,
                                  p1: float,
                                  p2: float) -> dict:
        """Return a dictionary of directions to a Dynamic Programming
        etimation of depth.

        For each direction in 1, ..., 8, calculate scores tensors
        according to dp_grade_slice and the method which allows you to
        extract slices along each direction.

        You may use helper methods (functions) that you write on your own.
        We found `np.diagonal` to be very helpful to extract diagonal slices.
        `np.unravel_index` might be helpful if you're thinking in MATLAB
        notations: it's the ind2sub equivalent.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for
            every pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.

        Returns:
            Dictionary int->np.ndarray which maps each direction to the
            corresponding dynamic programming estimation of depth based on
            that direction.
        """
        num_of_directions = 8
        l = np.zeros_like(ssdd_tensor)
        
        direction_to_slice = {}
        """INSERT YOUR CODE HERE"""
        for dir in range(1,num_of_directions+1):
            direction_to_slice[dir]= self.naive_labeling(self.extract_slices(ssdd_tensor,dir,p1,p1))
        return direction_to_slice

    def sgm_labeling(self, ssdd_tensor: np.ndarray, p1: float, p2: float):
        """Estimate the depth map according to the SGM algorithm.

        For each direction in 1, ..., 8, calculate scores tensors
        according to dp_grade_slice and the method which allows you to
        extract slices along each direction.

        You may use helper methods (functions) that you write on your own.
        We found `np.diagonal` to be very helpful to extract diagonal slices.
        `np.unravel_index` might be helpful if you're thinking in MATLAB
        notations: it's the ind2sub equivalent.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for
            every pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.

        Returns:
            Semi-Global Mapping depth estimation matrix of shape HxW.
        """
        num_of_directions = 8
        l = np.zeros_like(ssdd_tensor)
        #temp1= self.extract_slices(ssdd_tensor,8,p1,p1)
        """INSERT YOUR CODE HERE"""
        
        for dir in range(1,num_of_directions+1):
            temp = self.extract_slices(ssdd_tensor,dir,p1,p1)
            l+=temp
        l=(l/num_of_directions)
        
        return self.naive_labeling(l)
        #return self.naive_labeling(temp1)
