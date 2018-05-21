


%Function is reponsible for taking a filename and outputing the distances between all windows of the input and 
%all the chirplets in the deck. 

%Res_vec is the results vector, containing all dot products between different shifts of the input signal and the chirplets. 

function [res_vec] = dist_window_approach(filename,all_chirps_cell,deck_param,draw_parameters,window_length_ts)

[y,f] = audioread(filename);


sample_step = deck_param.time_sample_step;

pad_value = max(window_length_ts);


y_pad = [zeros(1,pad_value) y' zeros(1,pad_value)];

start = pad_value + 1;
stop = length(y_pad) - pad_value;


results_cell = cell(1,length(window_length_ts));


res_vec = [];


energies = [];




for k = 1:length(all_chirps_cell)
    
        num_chirps = length(all_chirps_cell{k}(:,1));
        chirp_l = length(all_chirps_cell{k}(1,:));
        chirp_l_2 = chirp_l / 2;
        test_vector = [start:sample_step:stop];
        test_number = length(test_vector);
        
        x_vector = zeros(chirp_l,test_number);
        x_vector_e = zeros(test_number,num_chirps);
        o = 1;
        for p = start:sample_step:stop
            
            index = p;
            sample = y_pad(floor(p-chirp_l_2+1):floor(p+chirp_l_2));

            x_vector(:,o) = sample;
            x_vector_e(o,:) = sum(abs(sample).^2) / length(sample) ;
            energies(k,o) = sum(abs(sample).^2) / length(sample) ;
           %x_vector_e(o,:) = sum(abs(sample).^2) / length(sample);
            o = o + 1;
         
        end
        
        %Calculate energy sum(abs(x).^2) for each x_vector;
        %Multiply the res_vec by it.
        %This is the Pearson Correlation Coeffecient being used. 
        temp = corr(all_chirps_cell{k}',x_vector)';
        
        if draw_parameters.en_cor == 1 
            temp = temp .* x_vector_e;
        end
        
        res_vec = [res_vec temp];
        

end


