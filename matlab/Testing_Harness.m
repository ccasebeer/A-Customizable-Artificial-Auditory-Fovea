%NN_testing_harness.

%This script runs through the mat_test_directory looking for test setups. 
%The test setup specifies if this is a spectrogram or a chirpletgram method. 
%If its a spectrogram, what window sizes should be used. 
%If its a chirpletgram, what deck parmaters should be used. 


%By the end of the script, a corrpesponding .mat file will exist in
%mat_test_results for each test setup.
%which can be trained on using the Python scripts.


set_directories

mat_test_dir = directory_setup.mat_test_dir;
mat_test_results_dir = directory_setup.mat_test_results_dir;
test_data_directory = directory_setup.test_data_directory;
working_dir = directory_setup.working_dir;


cd (mat_test_dir);

addpath(working_dir)


cur_dir = pwd
dinfo = dir
names_cell =  {dinfo.name};

names_cell(1:2) = [];

%Other parameters for drawing or matching.
draw_parameters.transparency_en = 0;
draw_parameters.en_cor = 0;
draw_parameters.scatter_en = 0;
draw_parameters.index_thold = .9;
draw_parameters.log_en = 0; 


%Filename count index
count = 1;
for k = 1:length(names_cell)
   
    
    cd (mat_test_dir);
    
    if strcmp(names_cell{k},'old')
        continue
    end
    
    load(names_cell{k});

    sample_step = deck_param.time_sample_step;

    %function [all_chirps all_desc freq_slopes t window_length_ts chirp_type_key] = chirplets_f(deck_param)

    [all_chirps all_desc freq_slopes t window_length_ts chirp_type_key deck_param] =  chirplets_f(deck_param);
    
    %function [results_cell results_flat_padded results_spec results_flat_spec_padded y] = prep_data_f(chirp_on, spec_on, spec_window_size,spec_window_overlap fft_size, all_chirps,freq_slopes, all_desc, t)

    [results_cell  results_spec  y total_cell total_names_cell] =  Prep_Data_f(chirp_en, spec_en, spec_window_size, spec_window_overlap ,spec_type,fft_size, all_chirps, freq_slopes,all_desc, t, window_length_ts, sample_step,directory_setup,deck_param,draw_parameters);

   
   
   create_4D_decks
   
   
end