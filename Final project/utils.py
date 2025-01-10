import numpy as np
import pandas as pd
import SimpleITK as sitk
import os
import time
from intensity_normalization.plot.histogram import HistogramPlotter
from matplotlib import pyplot as plt

def read_nifti_file(filepath):
    """Read a NIfTI file and return the image data, origin, and spacing.

    Args:
        filepath (str): Path to the NIfTI file.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The CT scan as a numpy array.
            - np.ndarray: The origin of the image.
            - np.ndarray: The spacing of the image.
    """
    itkimage = sitk.ReadImage(filepath)
    ct_scan = sitk.GetArrayFromImage(itkimage)  # Convert to numpy array (z, y, x)
    origin = np.array(list(reversed(itkimage.GetOrigin())))  # Reverse to get (x, y, z)
    spacing = np.array(list(reversed(itkimage.GetSpacing())))  # Reverse to get (x, y, z)
    return ct_scan, origin, spacing

def save_as_nifti(array, filename, reference_image):
    """Save array as nifti image

    Args:
        array (array): array to be saved
        filename (str): path to save
        reference_image (str): path of reference image
    """
    # Read the reference image and create a new image from the array
    reference_image = sitk.ReadImage(reference_image)
    image = sitk.GetImageFromArray(array)

    # Set the properties of the new image based on the reference image
    image.CopyInformation(reference_image)

    # Write the new image to the specified filename
    sitk.WriteImage(image, filename)
    
def get_landmark(phase, patient_num):
    """Get the landmarks of the patient based on inhale or exhale.

    Args:
        phase (str): Inhale ('i') or exhale ('e').
        patient_num (int): Patient number.

    Returns:
        np.ndarray: The landmarks as a numpy array scaled by the image spacing.
    """
    image_path = f"data/copd{patient_num}/copd{patient_num}_{phase}BHCT.nii.gz"
    landmark_file = f"data/copd{patient_num}/copd{patient_num}_300_{phase}BH_xyz_r1.txt"
    
    image = sitk.ReadImage(image_path)
    spacing = np.array(image.GetSpacing())  # Get spacing (x, y, z)
    
    # Read landmarks from the file
    landmark = pd.read_csv(landmark_file, sep="\t" if phase=='i' else "\t ", header=None, engine='python')
    
    # Scale landmarks by spacing to convert to mm
    return landmark.values * spacing  # Return as numpy array

def calculate_tre(fixed_points, moving_points):
    """Calculate the Target Registration Error (TRE) between fixed and moving points.

    Args:
        fixed_points (np.ndarray): The fixed points.
        moving_points (np.ndarray): The moving points.

    Returns:
        np.ndarray: The TRE for each point.
    """
    return np.sqrt(np.sum((fixed_points - moving_points) ** 2, axis=1))

def calculate_tre_mean_std(fixed_points, moving_points):
    """Calculate the mean and standard deviation of the Target Registration Error (TRE).

    Args:
        fixed_points (np.ndarray): The fixed points.
        moving_points (np.ndarray): The moving points.

    Returns:
        tuple: A tuple containing:
            - float: The mean TRE.
            - float: The standard deviation of the TRE.
    """
    tre = calculate_tre(fixed_points, moving_points)
    return np.mean(tre), np.std(tre)

def visualize_histograms(images, title):
    """Visualize histograms of the given images."""
    hp = HistogramPlotter(title=title)
    _ = hp(images, masks=None)
    plt.show()

def read_parameter_file(file_path):
    return sitk.ReadParameterFile(file_path)

def multi_parameter_list(file_paths):
    parameter_list = []
    for file_path in file_paths:
        parameter_list.append(read_parameter_file(file_path))
    return parameter_list

def convert_txt_to_pts_with_header(txt_file_path):
    """Convert a TXT file containing point data to a PTS file format.

    Args:
        txt_file_path (str): Path to the input TXT file containing point data.

    The function reads the TXT file, counts the number of points, and writes
    the data to a new PTS file. The first line of the PTS file contains the
    number of points, followed by the point data with tabs replaced by spaces.
    """
    # Open the TXT file to read data
    with open(txt_file_path, 'r') as file:
        data = file.readlines()

    # Count the number of points
    number_of_points = len(data)
    pts_file_path = txt_file_path.replace('.txt', '.pts')
    # Open or create a PTS file to write the converted data
    with open(pts_file_path, 'w') as pts_file:
        # Write the number of points as the header
        pts_file.write(f"{number_of_points}\n")
        for line in data:
            # Clean up the line, replace tabs with spaces, and write to the PTS file
            formatted_line = line.replace('\t', ' ').strip()
            pts_file.write(formatted_line + '\n')

def process_outputpoints(points_name, patient_num, save_files=True):
    """Read the output file and return the coordinates of the points
       for TRE Calculation.

    Args:
        points_name (str): Path to the outputpoints.txt file.
        patient_num (int): Patient number for file path construction.
        save_files (bool): Whether to save the coordinates to a file. Default is True.

    Returns:
        np.ndarray: Array with the coordinates of the points, or None if an error occurs.
    """ 
    try:
        df = pd.read_csv(f'data/copd{patient_num}/{points_name}', sep="\t", header=None)
    except FileNotFoundError:
        print(f"Error: The file {points_name} was not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    OUTPUT_POINTS_COLUMN = 5
    FINAL_COORD_COLUMN = 4

    # Check if the required columns exist
    if df.shape[1] <= max(OUTPUT_POINTS_COLUMN, FINAL_COORD_COLUMN):
        print("Error: The expected columns are not present in the data.")
        return None

    # Extract and clean output points
    outpoints = df.iloc[:, OUTPUT_POINTS_COLUMN].str.extract(r'OutputPoint = \[ ([\d\s\.\-]+) \]')[0]
    outpoints = outpoints.str.split(' ', expand=True).astype(float).abs().to_numpy()
    
    
    
    if save_files:
        # Save final coordinates to a text file
        final_coord = df.iloc[:, FINAL_COORD_COLUMN].str.extract(r'OutputIndexFixed = \[ ([\d\s]+) \]')[0]
        final_coord = final_coord.str.split(' ', expand=True).astype(int)
        np.savetxt(f"data/copd{patient_num}/outputpoints_coord_{points_name}", final_coord.to_numpy(), fmt='%s', delimiter='\t')
        # Save outpoints array in mm to a text file
        np.savetxt(f"data/copd{patient_num}/outputpoints_mm_{points_name}", outpoints, fmt='%.6f', delimiter='\t')
    return outpoints

def register_images(fixed_image_path, moving_image_path, fixed_landmarks, moving_landmarks,  fixed_mask_path, moving_mask_path, 
                     parameter_files, patient_num, param='default_affine', mask=False, suffix="param_11"):
    start_time = time.time()
    fixed_image = sitk.ReadImage(fixed_image_path)
    moving_image = sitk.ReadImage(moving_image_path)

    if mask:
        fixed_label = sitk.ReadImage(fixed_mask_path)
        moving_label = sitk.ReadImage(moving_mask_path)
        fixed_label.SetSpacing(fixed_image.GetSpacing())
        moving_label.SetSpacing(moving_image.GetSpacing())

    # Start registration settings
    elastixImageFilter = sitk.ElastixImageFilter()
    
    # Set parameter file or create default parameter maps
    parameterMapVector = sitk.VectorOfParameterMap()
    if parameter_files:
        for param_file in parameter_files:
            parameterMapVector.append(param_file)
    else:
        if param == 'default_affine':
            parameter_map_affine = sitk.GetDefaultParameterMap("affine")
            parameterMapVector.append(parameter_map_affine)
        elif param == 'default_bspline':
            parameter_map_bspline = sitk.GetDefaultParameterMap("bspline")
            parameterMapVector.append(parameter_map_bspline)
        else:
            print("Error: Invalid parameter option")
            return None

    # Set the parameter maps to the elastix filter
    elastixImageFilter.SetParameterMap(parameterMapVector)

    # Set fixed and moving images
    elastixImageFilter.SetFixedImage(fixed_image)
    elastixImageFilter.SetMovingImage(moving_image)
    
    if mask:
        elastixImageFilter.SetFixedMask(fixed_label)
        elastixImageFilter.SetMovingMask(moving_label)
    
    # Execute the registration
    elastixImageFilter.Execute()
    registered_image = elastixImageFilter.GetResultImage()
    
    # Save the transformed image
    param_file_name = suffix
    mask_status = "with_mask" if mask else "without_mask"
    registered_image_path = f"data/copd{patient_num}/registered_image_{param_file_name}_{mask_status}_{os.path.basename(fixed_image_path)}_{os.path.basename(moving_image_path)}.nii"
    sitk.WriteImage(registered_image, registered_image_path)
    
    # Get the transformation parameters
    transform_parameter_map = elastixImageFilter.GetTransformParameterMap()
    
    # Transform the landmarks
    transformixImageFilter = sitk.TransformixImageFilter()
    transformixImageFilter.SetTransformParameterMap(transform_parameter_map)
    transformixImageFilter.SetFixedPointSetFileName(str(fixed_landmarks))
    transformixImageFilter.SetMovingImage(moving_image)  # Set moving image (needed to get spatial information)
    transformixImageFilter.Execute()
    
    # Get transformed points
    points_name = f"transformed_points_{param_file_name}_{mask_status}_{os.path.basename(fixed_landmarks)}_{os.path.basename(moving_landmarks)}.txt"
    outputpoints_path = 'outputpoints.txt'
    
    if os.path.exists(outputpoints_path):
        os.replace(outputpoints_path, f'data/copd{patient_num}/{points_name}')
    else:
        print(f"Warning: {outputpoints_path} does not exist. Cannot rename.")
        return registered_image, None  # Return early if the output points file does not exist
    
    processed_points = process_outputpoints(points_name, patient_num, True)

    # Compute the TRE
    gt_landmarks = get_landmark(patient_num=patient_num, phase='e')
    tre_mean, tre_std = calculate_tre_mean_std(gt_landmarks, processed_points)
    print(f'TRE mean: {tre_mean} mm, std: {tre_std} mm')
    end_time = time.time()  # End tracking time
    computation_time = end_time - start_time
    print(f"Computation time: {computation_time:.2f} seconds")
    return registered_image, processed_points

def register_test_images(fixed_image_path, moving_image_path, fixed_landmarks, fixed_mask_path, moving_mask_path,  parameter_files,
                         patient_num, param='default_affine', mask=False, suffix="param_11"):
    start_time = time.time()
    fixed_image = sitk.ReadImage(fixed_image_path)
    moving_image = sitk.ReadImage(moving_image_path)

    if mask:
        fixed_label = sitk.ReadImage(fixed_mask_path)
        moving_label = sitk.ReadImage(moving_mask_path)
        fixed_label.SetSpacing(fixed_image.GetSpacing())
        moving_label.SetSpacing(moving_image.GetSpacing())

    # Start registration settings
    elastixImageFilter = sitk.ElastixImageFilter()
    
    # Set parameter file or create default parameter maps
    parameterMapVector = sitk.VectorOfParameterMap()
    if parameter_files:
        for param_file in parameter_files:
            parameterMapVector.append(param_file)
    else:
        if param == 'default_affine':
            parameter_map_affine = sitk.GetDefaultParameterMap("affine")
            parameterMapVector.append(parameter_map_affine)
        elif param == 'default_bspline':
            parameter_map_bspline = sitk.GetDefaultParameterMap("bspline")
            parameterMapVector.append(parameter_map_bspline)
        else:
            print("Error: Invalid parameter option")
            return None

    # Set the parameter maps to the elastix filter
    elastixImageFilter.SetParameterMap(parameterMapVector)

    # Set fixed and moving images
    elastixImageFilter.SetFixedImage(fixed_image)
    elastixImageFilter.SetMovingImage(moving_image)
    
    if mask:
        elastixImageFilter.SetFixedMask(fixed_label)
        elastixImageFilter.SetMovingMask(moving_label)
    
    # Execute the registration
    elastixImageFilter.Execute()
    registered_image = elastixImageFilter.GetResultImage()
    
    # Save the transformed image
    param_file_name = suffix
    mask_status = "with_mask" if mask else "without_mask"
    registered_image_path = f"data/copd{patient_num}/registered_image_{param_file_name}_{mask_status}_{os.path.basename(fixed_image_path)}_{os.path.basename(moving_image_path)}.nii"
    sitk.WriteImage(registered_image, registered_image_path)
    
    # Get the transformation parameters
    transform_parameter_map = elastixImageFilter.GetTransformParameterMap()
    
    # Transform the landmarks
    transformixImageFilter = sitk.TransformixImageFilter()
    transformixImageFilter.SetTransformParameterMap(transform_parameter_map)
    transformixImageFilter.SetFixedPointSetFileName(str(fixed_landmarks))
    transformixImageFilter.SetMovingImage(moving_image)  # Set moving image (needed to get spatial information)
    transformixImageFilter.Execute()
    
    # Get transformed points
    points_name = f"transformed_points_{param_file_name}_{mask_status}_{os.path.basename(fixed_landmarks)}.txt"
    outputpoints_path = 'outputpoints.txt'
    
    if os.path.exists(outputpoints_path):
        os.replace(outputpoints_path, f'data/copd{patient_num}/{points_name}')
    else:
        print(f"Warning: {outputpoints_path} does not exist. Cannot rename.")
        return registered_image, None  # Return early if the output points file does not exist
    
    processed_points = process_outputpoints(points_name, patient_num, True)

    end_time = time.time()  # End tracking time
    computation_time = end_time - start_time
    print(f"Computation time: {computation_time:.2f} seconds")
    return registered_image, processed_points