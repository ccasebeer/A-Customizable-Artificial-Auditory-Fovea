%Set Directories Script.

%Directory which holds the processing setup files.
directory_setup.mat_test_dir = 'F:\tensorflow_temp\mat_test_files'

%Directory where either the processed spectrogram data or the chirplet data is
%processed to. Tensorflow/Python reads this data. 
directory_setup.mat_test_results_dir = 'F:\tensorflow_temp\mat_test_results'

%Test set data. Classes organized by folder name. 
directory_setup.test_data_directory = 'G:\Users\Chris\GDrive\Research (1)\chirping\master_directory\test_data'

%Directory where Tensorflow will dump its data. 
%This directory will get occasionally very very large. 
directory_setup.tf_output_dir = 'F:\tensorflow_temp'

directory_setup.working_dir = pwd;

%Python will read this structure too. 
save('directory_setup.mat');
