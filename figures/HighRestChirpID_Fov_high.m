

%Generate three different sloping chirplets for figure. 

clear all;

close all


addpath('../matlab')

twitter = 'demo_dir/Twitter/A3910C20.580.wav'
phee = 'demo_dir/Phee/A261CC21.442.wav'
trill = 'demo_dir/Trill/C3260141.082.wav'
test = 'demo_dir/chirp_test.wav'


names = {twitter,phee,trill,test}


[y_read{1} f] = audioread(twitter);
[y_read{2} f] = audioread(phee);
[y_read{3} f] = audioread(trill);
[y_read{4} f] = audioread(test);


choice = 3;

y = y_read{choice};
spec_y =  y_read{choice};


deck_param.window_length_s = [.01];
deck_param.bottom_freq = 500;
deck_param.top_freq = 20000;
deck_param.center_point_step = 500;
% deck_param.freq_slopes_start = 100;
% deck_param.freq_slopes_stop = 1100;
% deck_param.freq_slopes_step = 500;
deck_param.freq_slopes_start = 10000;
deck_param.freq_slopes_stop = 250000;
deck_param.freq_slopes_step = 25000;
deck_param.time_sample_step = 50;
deck_param.f = f;
deck_param.num_chirps = [];
deck_param.chirp_amp_mod_key = {1,'gauss'}


%Add additional frequency bands of interest. "Foveation"
deck_param.freqs_of_interest = [6000:10:8000];

%deck_param.freqs_of_interest = [];


%Other parameters for drawing or matching.
draw_parameters.transparency_en = 0;
draw_parameters.en_cor = 0;
draw_parameters.scatter_en = 0;
draw_parameters.index_thold = .95;
draw_parameters.log_en = 0; 


sample_step = deck_param.time_sample_step;


[all_chirps all_desc freq_slopes t window_length_ts chirp_type_key deck_param] =  chirplets_f(deck_param);

results =  dist_window_approach_f(names{choice},all_chirps, deck_param,draw_parameters,window_length_ts);





%Mind the max value of each column of results 
max_result = [];

for k = 1:length(results(:,1))

    
[max_v index] = max(results(k,:));

    max_result(k,1) = index;
    max_result(k,2) = max_v;
end

%Remove negative chirplet matches. 
results(results<0) = 0;



figure;
graph_types = [2 4 5];
set(gcf, 'Position', [601 99 1209 1233]);
draw_plots_flat_all


saveas(gcf,'figures/HighResChirpID_fov_high.png');

axes(axes_array(end))
xlim([0.1137    0.3131])
ylim([5.3203    9.3080])


saveas(gcf,'figures/HighResChirpID_fov_zoom_high.png');




