%This function takes the chirplets as defined and then proceeds to generate resultant vector for
%all exemplar data. Two types of data are created, spectrograms and chirpletgrams. 

%Results_cell contains all resultant chirplet data across the exemplar data. 
%Results_spec tontains all resultant spectrogram data across all exemplar data.




function [results_cell results_spec y total_cell total_names_cell] = prep_data_f(chirp_on, spec_on, spec_window_size,spec_window_overlap,spec_type, fft_size, all_chirps,freq_slopes, all_desc, t ,window_length_ts, sample_step,directory_setup,deck_param,draw_parameters)



mat_test_dir = directory_setup.mat_test_dir;
mat_test_results_dir = directory_setup.mat_test_results_dir;
test_data_directory = directory_setup.test_data_directory;
working_dir = directory_setup.working_dir;

cd(working_dir)
cd(test_data_directory)

cur_dir = pwd
dinfo = dir
names_cell =  {dinfo.name};
names_cell(1:2) = [];

total_cell = cell(1);
total_names_cell = cell(1);

call_type = 0;

y = [];

codebook_names = cell(1);
codebook_codes = [];

%Filename count index
count = 1;
for k = 1:length(names_cell)
    
    cd (names_cell{k})
    dinfo = dir
    files_cell =  {dinfo.name};
    files_cell(1:2) = [];
    
    
    
    
    for l = 1:length(files_cell)
        if ismac
        total_cell{k}{l} = [cur_dir '/' names_cell{k} '/' files_cell{l}]
        else
        total_cell{k}{l} = [cur_dir '\' names_cell{k} '\' files_cell{l}]
        end
        y = [y call_type];
        
        total_names_cell{count} = total_cell{k}{l};
        count =count + 1;
    end
    
 
    codebook_names{k} = names_cell{k} ;
    codebook_codes(k) = call_type;  
    call_type = call_type + 1;
    
    cd ..
end


save([working_dir '\codebook.mat'], 'codebook_names', 'codebook_codes', 'y');



results_flat_padded = [];
results_flat_spec_padded = [];



results_cell = [];
results_spec = [];




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Chirplet methods.

if (chirp_on == 1)
    


    results_cell = cell(1);
    i = 1;
    for k = 1:length(total_cell)
        for p = 1:length(total_cell{k})


   % results_cell{i} =  dist_window_approach_f(total_cell{k}{p},all_chirps,t , 0, window_length_ts, sample_step);
    results_cell{i} =  dist_window_approach_f(total_cell{k}{p},all_chirps,deck_param,draw_parameters,window_length_ts);

    i = i +1

        end
    end




end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Spectrogram methods.


if (spec_on == 1)

results_spec = cell(1);
i = 1;



  for k = 1:length(total_cell)
      for p = 1:length(total_cell{k})



       filename = total_cell{k}{p};
       
       
       [y_data,f] = audioread(filename);
      

     
        %Here I am reducing size of wideband spectrogram. 
        narrow_size = .05;
        num_points = length(y_data) / (narrow_size*f);
           
           
        window_samples = spec_window_size*f;	% window samps
        noverlap = spec_window_overlap*f;
          
           

        [s,f,t] = spectrogram(y_data,floor(window_samples),floor(noverlap),fft_size,f,'yaxis');
        %result = fft(y_data,fft_size);

        %Reduce the number of data points to that of the narrowband version. 
        %The reasoning for this is that the data becomes too big. 
           if strcmp(spec_type,'narrow')
                    
           else
               
            new_stride = floor(length(t)/num_points);

            s = s(:,1:new_stride:end);
               
           end



        s = abs(s');

        results_spec{i} = s;


        i = i +1


      end
  end


end




