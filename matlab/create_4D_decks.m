
%This script takes the spectrogram data or the chirplet gram data from 
%prep_data_f and creates the 4D data types needed for python/tensorflow. 
%This script saves the data which Python and Tensorflow will read for training. 



%Reorder the results_cell into a 4D image
%(:,:,n,p) where p is the training image number.
% n is the chirplet type for each training sound. 

%Find the dimensions across the training set. 

num = length(freq_slopes);
chirp_types = [-num:1:num];

array = [];
full_array = cell(1);
freqs = [];
num = 1;

dim3_track = [];

%Here I am building my 4D convnet chirplet array. 
%Dimension 1 is the time axis.
%Dimension 2 is the frequency axis
%Dimension 3 is the chirplet type axis. 

%I have to permute axis 1 and 3 inside the loop. 
%This is as the convent takes color/depth in its 3 dimension, whereas
%this script puts it first at least initially. 

%full_array is now a cell array of jagged 3D chirplet images. 
%I then have to imresize into a non-jagged array, full_array_nc.
%This array has a 4 dimension, training number.


single_chirp_en = 0;

max_operator_en = 1;

chirplet_pick = 2;


save_images_on = 0;




% %Directories to save spectrogram images. 
% image_save_directory_orig = [image_save_dir 'orig\']
% image_save_directory = [image_save_dir 'resize\']


load ([working_dir '\codebook.mat'])



%Process the chirp results into tensorflow chirp 4D.

if (chirp_en == 1)
    
    

for p = 1:length(results_cell)
    results = results_cell{p};
    array = [];
    for h = 1:length(chirp_type_key)
        chirp_type = chirp_type_key(h);
        num = 1;
        for k = 1:length(all_desc(:,1))

            if (all_desc(k,1) == chirp_type)
                row = k;
                freq = all_desc(k,2);


            %Extract result row. 
            temp = results(:,row);


            array(h,num,:) = temp;
            
            freqs(num) = freq;
            
            
            num = num + 1;        

            end

        end
  
    end
    %Here a full chirp deck is complete.
    
    dim3_track = [dim3_track length(array(1,1,:))]
    


    %Change the order of the array to match the way tensorflow expects it. 
    %Tensor input become 4-D: [Exemplar, Height, Width, Channel]

    array = permute(array,[3 2 1]);
    full_array{p} = array;
    
    
end

min_x_dim = min(dim3_track);
dim_y = length(array(1,:,1));




if (single_chirp_en == 1)
   full_array_nc = zeros(min_x_dim,length(array(1,:,1)),1,length(full_array)); 
else
   full_array_nc = zeros(min_x_dim,length(array(1,:,1)),length(array(1,1,:)),length(full_array));
end





%Create full_array_nc (the resized array) from full_array (the jagged array)
%This is done with imresize. 

for k = 1:length(full_array)
    
    temp = []
    temp2 = []
    temp2p = []
    temp = full_array{k};
    
    
    temp = max(temp,[],3);
    
    %Sanity check. Flatten the array and display before and after resize. 
    
    
    temp2 = imresize(full_array{k}, [min_x_dim dim_y []]);
    
    %Uncomment to see before and after resizing on max_chirplet_flat
%     subplot(2,1,1)
%     imagesc(temp)
%     subplot(2,1,2)
%    temp2p = max(temp2,[],3);
%     imagesc(temp2p)
    
    
    if (single_chirp_en == 1)
        
        temp = full_array{k};
        temp = temp(:,:,chirplet_pick);
        temp2 = imresize(temp, [min_x_dim dim_y]);
        full_array_nc(:,:,1,k) = temp2;   
    else
        
        full_array_nc(:,:,:,k) = temp2;
    end

    
end

%If the max operator is on, sparsify the decks.
%Select the greatest point across the chirp types. Zero all other types. 
if ( max_operator_en == 1)

    
dim1 = length(full_array_nc(:,1,1,1));
dim2 = length(full_array_nc(1,:,1,1));
dim3 = length(full_array_nc(1,1,:,1));
dim4 = length(full_array_nc(1,1,1,:));
    
     %This is done with a reshape, max with indices.
     %Extract examplar scan
     %Extract all deck scan for that exemplar. 
     %Reshape and take max with index. 
     %Create full zeroed. Fill in the best chirplets. 
     %Reshape back.
    for k = 1:length(full_array_nc(1,1,1,:))
            temp = full_array_nc(:,:,:,k);
            temp2 = reshape(temp,[], dim3);

            [m i] = max(temp2,[],2);
            temp3 = zeros(dim1*dim2,dim3);
            p = 1;
            for l = 1:length(i)
               temp3(p,i(l)) = m(l);
               p = p + 1;
            end

            temp4 = reshape(temp3,dim1,dim2,dim3);
            full_array_nc(:,:,:,k) = temp4;

    end
    
end


length(full_array_nc(:,1,1,1))
length(full_array_nc(1,:,1,1))
length(full_array_nc(1,1,:,1))
length(full_array_nc(1,1,1,:))


%Save out the 4D array for import into tensorflow. 
% temp_dir = 'G:\Users\Chris\GDrive\Research (1)\chirping\Tensorflow_Playground\TensorFlow-Examples\notebooks\3_NeuralNetworks'
% save([temp_dir '/chirp_4D.mat'], 'full_array_nc', 'min_x_dim', 'dim_y', 'y', 'freq_slopes', 'all_desc', 'window_length_s', 'bottom_freq', 'top_freq', 'center_point_step', 'freq_slopes_step', 'freq_slopes_start', 'freq_slopes_stop', 'slopes_of_interest', 'bands_of_interest', 'test_description');

save([mat_test_results_dir '/' test_description '_r.mat'], 'full_array_nc', 'min_x_dim', 'dim_y', 'y', 'deck_param', 'all_desc', 'test_description');

end





%Process the spectrogram methods into tensorflow spec 4D.

if (spec_en == 1)


%Take in spectrogram results.
%Create 4D array for the convnet.
%Resize the spectrograms to minumum dimensions (time axis shrink)
%The frequency axis stays constant due to constant fft length. 
test = results_spec;

dimensions = zeros(length(test),2);
for k = 1:length(test)
    
    dimensions(k,:) = [length(test{k}(1,:)) length(test{k}(:,1))]  ;

end

%Resize all images (spectrograms) to the minimum size. 
%These will be fed to the covnet for classification. 


use_max_dim_for_resize = 0;

if (use_max_dim_for_resize == 1)
    x_ord = max(dimensions(:,1))
    y_ord = max(dimensions(:,2))
else
    x_ord = min(dimensions(:,1))
    y_ord = min(dimensions(:,2))
end



test_data = zeros(x_ord,y_ord,1,length(test));

for k = 1:length(test)
    
    imagesc(test{k});
    imagesc(imresize(test{k},[x_ord y_ord]));
    
    test_data(:,:,1,k) = imresize(test{k},[x_ord y_ord]);
    
end


spec_4d = test_data;


% if(save_images_on == 1)
%     
%     for k = 1:length(spec_4d(1,1,1,:))
%         
%         
%         type = y(k);
%         name = codebook_names{type+1};
%         
%         A = results_spec{k};
%         A=A-min(A(:)); % shift data such that the smallest element of A is 0
%         A=A/max(A(:)); % normalize the shifted data to 1 
%         
%         
%         
%         imwrite(A,[image_save_directory_orig name '_' num2str(k) '.png'])
%         
%         A = spec_4d(:,:,1,k);
%         A=A-min(A(:)); % shift data such that the smallest element of A is 0
%         A=A/max(A(:)); % normalize the shifted data to 1 
%         
%         
%         
%         imwrite(A,[image_save_directory name '_' num2str(k) '.png'])
%         
%         
%         
% 
%     end
%     
% end


length(spec_4d(:,1,1,1))
length(spec_4d(1,:,1,1))
length(spec_4d(1,1,:,1))
length(spec_4d(1,1,1,:))



% temp_dir = 'G:\Users\Chris\GDrive\Research (1)\chirping\Tensorflow_Playground\TensorFlow-Examples\notebooks\3_NeuralNetworks'
% save([temp_dir '/spec_4D.mat'], 'spec_4d', 'x_ord', 'y_ord', 'y', 'spec_window_size', 'spec_window_overlap','fft_size');
save([mat_test_results_dir '/' test_description '_r.mat'], 'spec_4d', 'x_ord', 'y_ord', 'y', 'spec_window_size', 'spec_window_overlap','fft_size', 'test_description');


end


