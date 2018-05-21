

%Create a setup for processing the example data. 


%Choose either a spectrogram OR a chirpletgram.
%Choose spectrogram by setting window sizing and setting enable.
%Choose chirpletgram by 



clear all
mat_test_directory = 'F:\tensorflow_temp\mat_test_files'


f = 50000;

deck_param.window_length_s = [.005 .01];
deck_param.bottom_freq = 500;
deck_param.top_freq = 20000;
deck_param.center_point_step = 1000;
deck_param.freq_slopes_start = 100;
deck_param.freq_slopes_stop = 220000;
deck_param.freq_slopes_step = 26250;
deck_param.time_sample_step = 2000;
deck_param.chirp_amp_mod_key = {1,'gauss'}
deck_param.f = f;
deck_param.num_chirps = [];




% % Long window in time - Narrowband
spec_window_size = .05;
spec_window_overlap = 0;
fft_size = 1024;


% % % Spectrogram, Balanced,  window in time
% spec_window_size = .005;
% spec_window_overlap = 0;
% fft_size = 1024;

% % % Spectrogram, Wide,  window in time
% spec_window_size = .0005;
% spec_window_overlap = 0;
% fft_size = 1024;


%Specify narrow or wide. 
%Narrow spectrograms are subsampled
spec_type = 'narrow';



%Choose either/or. 
chirp_en = 0;
spec_en = 1;

formatOut = 'mm_dd_yy';
date_str = datestr(now,formatOut)

%Put either 'Spec' or 'Chirp' in the description for things to work. 
test_description = ['Spec_narrow_' date_str];


test_description = strrep(test_description, ' ', '_');

save([mat_test_directory '\' test_description '.mat']);