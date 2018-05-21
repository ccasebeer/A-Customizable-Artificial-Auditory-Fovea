

function [all_chirps all_desc freq_slopes t window_length_ts chirp_type_key deck_param] = chirplets_f(deck_param)




%Deck_Param
%Structure defining the chirplet deck.
%Number of chirp types at each point is filled in.

%all_chirps
%The actual chirps.
%N time cell arrays of all the actual chirplet time waveforms. 


%all_desc
%Description of every chirplet waveform
%Columns as follows. 
% 1 Unique Identifier based on slope and length but not frequency location
% 2 Center Frequency
% 3 Defined Slopes
% 4 Frequency Start
% 5 Frequency End
% 6 Length in Samples at the Given Sampling Rate
% 7 Calculated Slope (Double check the slopes we defined is the actual slopes)
% 8 Length Type (1:N lengths) 
% 9 Amp Type (1:N amp types)


%chirp_type_key
% Is a series of columsn which describe each type of distinct chirplet (shape and length not frequency location)
% The rows are defined as follows
% 1 Unique Identifier
% 2 Length Type (1:N lengths) 
% 3 Slope
% 4 Amp Type (1:N amp types)
% 5-7 RGB colors assigned to THIS chirp type. Used later for ID plots and chirplet drawing. 


%freq_slopes
%Vector of the unique slopes across all chirplet types. 

%window_length_ts
%Length of the chirplets in samples, converted from the deck_param specification of seconds, given Fs. 





window_length_s = deck_param.window_length_s;
bottom_freq  = deck_param.bottom_freq;
top_freq  = deck_param.top_freq;
center_point_step  = deck_param.center_point_step;
freq_slopes_step  = deck_param.freq_slopes_step;
freq_slopes_start  = deck_param.freq_slopes_start;
freq_slopes_stop  = deck_param.freq_slopes_stop;
chirp_amp_mod_key = deck_param.chirp_amp_mod_key;


center_point_freq_add = [];
slopes_of_interest  = [];
bands_of_interest  = [];


if (~isempty(bands_of_interest))
for k = 1:length(bands_of_interest(:,1))
    center_point_freq_add = [center_point_freq_add bands_of_interest(k,:)];
end
end


window_length_ts = [];


%f = 50000;
f = deck_param.f;


dur_band_c = [];
dur_band_u = [];
dur_band_d = [];


c_results = [];
u_results = [];
d_results = [];
c_chirps = [];
u_chirps = [];
d_chirps = [];
f_span = 500;
f_space = 100;


i=1;




c_array = [];





center_point_freq = [bottom_freq:center_point_step:top_freq];

center_point_freq = [center_point_freq center_point_freq_add];


%Sort ascending so plot methods will still work.
%Remove duplicates from base. 
center_point_freq = sort(center_point_freq);
center_point_freq = unique(center_point_freq);


% 
freq_slopes = [freq_slopes_start:freq_slopes_step:freq_slopes_stop];
freq_slopes = [freq_slopes slopes_of_interest];
freq_slopes = sort(freq_slopes);
freq_slopes = unique(freq_slopes);



num_chirp_types = length(freq_slopes)*2+1;

num_slopes = length(freq_slopes);


%Given a specific window length, generate a chirp of the given frequency
%slope. 
chirp_num = 0;
u = 1;
c = 1;


c_chirps_cell = cell(length(window_length_s),1);
d_chirps_cell  = cell(length(window_length_s),1);
u_chirps_cell  = cell(length(window_length_s),1);
all_chirps_cell  = cell(length(window_length_s),1);
all_desc_cell  = cell(length(window_length_s),1);
c_chirp_desc = [];
d_chirp_desc = [];
u_chirp_desc = [];

%Slope type + length;
chirp_type_key = [;];





%Are the other amp types enabled....
%This should be more elegant. 

if (length(deck_param.chirp_amp_mod_key(:,1)) == 1)

guass_only = 1;

else 
    
guass_only = 0;  

end





for w = 1:length(window_length_s)
   


    t=0:1/f:window_length_s(w);
    window_length_ts = [window_length_ts length(t)];
    chirp_length = length(t);
    c_chirps = [];
    d_chirps = [];
    u_chirps = [];
    
    c = 1;
    u = 1;
    
    slopes_created = [];

    for k = 1:length(center_point_freq)

        
        center = center_point_freq(k);
        
        
        
       	%Create guass_amp chirps of constant freq @ center_freq
        type = chirp_amp_mod_key{1,2};   
        desc = str2num([num2str((chirp_length + 0)) num2str(1)]);
       	[c_chirps(c,:) start stop slope_out] = gen_chirp_slope(center,0,t,f, type);  
        c_chirp_desc(c,:) = [desc center 0 start stop length(t) slope_out w chirp_amp_mod_key{1,1}];
        c = c + 1; 
        temp = [desc; w; 0 ; chirp_amp_mod_key{1,1} ];
        chirp_type_key = cat(2,chirp_type_key,temp);
        
        if (guass_only ~= 1)
            
            
            %Create up_amp chirps of constant freq  @ center_freq
            type = chirp_amp_mod_key{2,2};
            desc = str2num([num2str((chirp_length + 0)) num2str(2)]);
            [c_chirps(c,:) start stop slope_out] = gen_chirp_slope(center,0,t,f, type);  
            c_chirp_desc(c,:) = [desc center 0 start stop length(t) slope_out w chirp_amp_mod_key{2,1}];
            c = c + 1;
            temp = [desc; w ; 0 ; chirp_amp_mod_key{2,1} ];
            chirp_type_key = cat(2,chirp_type_key,temp);


            %Create down_amp chirps of constant freq  @ center_freq
            type = chirp_amp_mod_key{3,2};
            desc = str2num([num2str((chirp_length + 0)) num2str(3)]);
            [c_chirps(c,:) start stop slope_out] = gen_chirp_slope(center,0,t,f, type);  
            c_chirp_desc(c,:) = [desc center 0 start stop length(t) slope_out w chirp_amp_mod_key{3,1}];
            c = c + 1;
            temp = [desc; w; 0 ; chirp_amp_mod_key{3,1} ];
            chirp_type_key = cat(2,chirp_type_key,temp);
        
        end
        
        
        %chirp_type_key = [chirp_type_key(1,:) (chirp_length + 0) ; chirp_type_key(2,:) w ];

        
        for p = 1:length(freq_slopes)
            
            
            
            
            
            
            slope = freq_slopes(p);
            %Assemble up and down guass chirp of slope @ center
         	desc1 = str2num([num2str(((chirp_length-p))) num2str(1)]);
            desc2 = str2num([num2str(((chirp_length+p))) num2str(1)]);
            [d_chirps(u,:) start stop slope_out] = gen_chirp_slope(center,-slope,t,f,'gauss');
            d_chirp_desc(u,:) = [desc1 center -slope start stop length(t) slope_out w chirp_amp_mod_key{1,1}];
            [u_chirps(u,:) start stop slope_out]  = gen_chirp_slope(center,slope,t,f,'gauss');
            u_chirp_desc(u,:) = [desc2 center slope start stop length(t) slope_out w chirp_amp_mod_key{1,1}];
            u = u + 1;

            temp = [desc1 desc2; w w; 0 0;chirp_amp_mod_key{1,1} chirp_amp_mod_key{1,1} ];
        	chirp_type_key = cat(2,chirp_type_key,temp);
              
            if (guass_only ~= 1)
            
                %Assemble up and down up lin chirp of slope @ center
                slope = freq_slopes(p);
                type = chirp_amp_mod_key{2,2};
                desc1 = str2num([num2str(((chirp_length-p))) num2str(2)]);
                desc2 = str2num([num2str(((chirp_length+p))) num2str(2)]);
                [d_chirps(u,:) start stop slope_out] = gen_chirp_slope(center,-slope,t,f,type);
                d_chirp_desc(u,:) = [desc1 center -slope start stop length(t) slope_out w chirp_amp_mod_key{2,1}];
                [u_chirps(u,:) start stop slope_out]  = gen_chirp_slope(center,slope,t,f,type);
                u_chirp_desc(u,:) = [desc2 center slope start stop length(t) slope_out w chirp_amp_mod_key{2,1}];
                u = u + 1;

                temp = [desc1 desc2; w w; 0 0;chirp_amp_mod_key{2,1} chirp_amp_mod_key{2,1} ];
                chirp_type_key = cat(2,chirp_type_key,temp);


                %Assemble up and down up down chirp of slope @ center
                slope = freq_slopes(p);
                type = chirp_amp_mod_key{3,2};
                desc1 = str2num([num2str(((chirp_length-p))) num2str(3)]);
                desc2 = str2num([num2str(((chirp_length+p))) num2str(3)]);
                [d_chirps(u,:) start stop slope_out] = gen_chirp_slope(center,-slope,t,f,type);
                d_chirp_desc(u,:) = [desc1 center -slope start stop length(t) slope_out w chirp_amp_mod_key{3,1}];
                [u_chirps(u,:) start stop slope_out]  = gen_chirp_slope(center,slope,t,f,type);
                u_chirp_desc(u,:) = [desc2 center slope start stop length(t) slope_out w chirp_amp_mod_key{3,1}];
                u = u + 1;



                temp = [desc1 desc2; w w; 0 0;chirp_amp_mod_key{3,1} chirp_amp_mod_key{3,1} ];
                chirp_type_key = cat(2,chirp_type_key,temp);
              

            end
              
            

            
        end

    
    end

    c_chirps_cell{w} = c_chirps;
    d_chirps_cell{w} = d_chirps;
    u_chirps_cell{w} = u_chirps;
    all_chirps_cell{w} = [c_chirps;d_chirps;u_chirps];
    all_desc_cell{w} = [c_chirp_desc;d_chirp_desc;u_chirp_desc];
end

%Descptrion Vectors
%[chirp_type (



all_chirps = all_chirps_cell;


all_desc = [];



for k = 1:length(all_desc_cell)
    
    all_desc = [all_desc; all_desc_cell{k}];
    
end

chirp_type_key = unique(chirp_type_key','rows')';

deck_param.num_chirps = length(chirp_type_key(1,:));



%Add slope information to the chirp type key to help draw the chirplet deck with correct labels.
%Index into the all_desc structure which is chirp descriptions. 

%Get a 1D of chirp id's.
chirp_keys = all_desc(:,1); 
for k = 1:length(chirp_type_key)

    chirp_type = chirp_type_key(1,k);

    %Find first instance of that chirp_id in the all_desc
    temp = find(chirp_keys == chirp_type);
    %Pull slope out of all_desc and add to chirp_type_key.
    chirp_type_key(3,k) = all_desc(temp(1),3);

end



end







