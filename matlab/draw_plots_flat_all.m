
%draw_plots_flat_all.


%Multiple methods for examplar and results visualization. 
%Draws different kinds of spectrograms for input. 
%Draws the best chirplet max match images
%Draws the best chirplet indicator graph.



%Test scenarios should change supress_i = (values > .999999999) ;
%and
%scatter_en = 1 inside the program. This will test for exact matches
%and plot them individually via scatter.


%Christopher Casebeer. christopher.casebee1@msu.montana.edu 2018.


chirp_length = length(all_chirps(1,:));

%For a particular chirp type, extract 2d array



array = [];
freqs = [];
num = 1;


%Define spectrograms. 
%Long window in time - Narrowband
spec_window_size_n = .1;
spec_window_overlap = .001;
fft_size = 1024;

w_size_narrow = spec_window_size_n*f;
o_size_narrow = spec_window_overlap*f;
% Spectrogram, Balanced, medium window in time
spec_window_size_b = .005;
spec_window_overlap = 0;
fft_size = 1024;
w_size_bal = spec_window_size_b*f;
o_size_bal = spec_window_overlap*f;

% Spectrogram, Wideband, short window in time
spec_window_size_w = .0005;
spec_window_overlap = 0;
fft_size = 1024;
w_size_wide = spec_window_size_w*f;
o_size_wide = spec_window_overlap*f;


%This code is creating a 3D image. Where the depth is chirp type and the
%image is the dot-product between the tiling of the time frequency domain
%with chirplets and the sound under test. 
%For each chirp_type look through the results for that chirp row, extract
%and append that chirplet slice to that chirplets image.
%Do the same for each chirplet type. 
%array is a (n-chirp,n-center_point_freq,n-number_of_windows)
%array(1,:,:) should be chirplet type 1 chirplet_gram.
for k2 = 1:length(chirp_type_key)
    chirp_type = chirp_type_key(1,k2);
    num = 1;
    for k = 1:length(all_desc(:,1))

        if (all_desc(k,1) == chirp_type)
            row = k;
            freq = all_desc(k,2);


        %Extract result row. 
        temp = results(:,row);

        array(k2,num,:) = temp;
        freqs(num) = freq;

        num = num + 1;


        end


    end
end



%When I take the max I need to keep indices so that I can pull out chirp
%types.
%Extract out the locations of the max chirps. 
%Max across first dimension(chirp types) keeping index of max type.
[values indices] = max(array,[],1);
indices = squeeze(indices);




upper_f_limit = deck_param.top_freq;


%Supress all matches below a threshold.
%Set to .999999999 if you are looking for exact matches such as
%when running an exact search on test demo's.
%Otherwise set at ~.>1 to have readable data. 

%Should really be 1-(elf)

%Plot top N% of the matches only.

[f_1,x_1] =   ecdf(reshape(values,[],1));
a = find(f_1>draw_parameters.index_thold);
index = a(1);
threshold = x_1(index);



supress_i = (values > threshold) ;
%supress_i = (values > .999999999) ;
supress_i = squeeze(supress_i);
indices = indices .* supress_i;


if draw_parameters.log_en == 1
array(array < 0) = 1E-9;
array = 10*log10(array);
end
log_en = draw_parameters.log_en;




num_graphs = length(graph_types);
axes_array = zeros(num_graphs,1);
cur_plot = 1;



%Deteremine colormap once.
num_colors = length(chirp_type_key);
cmap = distinct_colors(num_colors);


%Put the cmap colors into the chirp_type_key 
%Use these later for drawing.
for k =1:length(cmap)
    chirp_type_key(5:7,k) = cmap(k,:);
end


%Assign 
up_color = hex2rgb('ffff00');
down_color =  hex2rgb('403075');
%up_color = hex2rgb('FFFFFF');
%down_color =  hex2rgb('FFFFFF');
guass_color =   hex2rgb('7B9F35');
colors = [guass_color;up_color;down_color];


%Find unique amp types.
num_slopes = length(unique(chirp_type_key(3,:)));
slopes = unique(chirp_type_key(3,:));
up_slopes = find(slopes >= 0)
down_slopes = find(slopes < 0)
%Interpolate the UpSlopes and the Down Slopes
%on half each of a colormap. 
slope_map_order = [slopes(down_slopes) slopes(up_slopes)];
slope_map = colormap(hsv(length(slope_map_order)));

colormap default


%Find unique amp types.
num_amps = length(unique(chirp_type_key(4,:)));

%Create a 2D grid of colors for both time lengths and amp types.

y_axis_cm = slope_map_order;
x_axis_a = [1:num_amps];
x_axis_t = deck_param.window_length_s;

color_grid_a = zeros(length(y_axis_cm),length(x_axis_a));
color_grid_t = zeros(length(y_axis_cm),length(x_axis_t));




%This is order according to chirp_type_key. 

%Cmap is built sequentially as the column value of the chirp_type_key
%is what is mapped. cmap(1) maps to chirp_type_key's(:,1) chirp. 
 

for k =1:length(chirp_type_key)
    type = chirp_type_key(4,k);
    
    cur_slope = chirp_type_key(3,k);
    cur_time_index = chirp_type_key(2,k); 
    
    index = find(slope_map_order == cur_slope);
    %Build 2 color matrices now. 
    index_m = find(y_axis_cm == cur_slope);
    color_grid_a(index_m,type) = k;
    color_grid_t(index_m,cur_time_index) = k;
    
    
    switch type

        case 1

            %Assign all amp type 1, one color. 
%              	chirp_type_key(5:7,k) = colors(type,:);
%                 cmap(k,:) =  colors(type,:);

            %Assign slopes of Guassian colors. 

           

            cmap(k,:) =  slope_map(index,:);
            chirp_type_key(5:7,k) = slope_map(index,:);
            



        case 2
            chirp_type_key(5:7,k) = colors(type,:);
            cmap(k,:) =  colors(type,:);
            

        case 3
            chirp_type_key(5:7,k) = colors(type,:);
            cmap(k,:) =  colors(type,:);

    end

end


% 
%         image(x_axis_a,y_axis_cm,color_grid_a);
%         ax = gcf;
%         colormap(ax,cmap);
%         %Use climit to get the mapping we want. 
%         %0 - num_colors/number of chirplet types. 
%         %Where are the current limits?
%         c1 = caxis;
%         caxis([1 length(cmap)])
%         xticks(x_axis_a);
%         yticks(y_axis_cm);
%         xlabel('Amp Types');
%         ylabel('Chirp Slopes');




%Hack to get this variable into private Matlab function rewrite
%getSTFTColumns.m
%getSTFTColumns has been rewritten to evaluate the STFT at the deck sample step. 
%Not rewriting/moving the entire spectrogram method. 
global sample_step;
global enable_custom_getSTFT;
enable_custom_getSTFT = 1;
sample_step = deck_param.time_sample_step;


grid_size = 5;

for k =1:length(graph_types)

    switch graph_types(k)


    case 1

        [axes_array(cur_plot) width_subplot_spectrogram]  = draw_narrow (num_graphs, cur_plot,spec_y,w_size_narrow,o_size_narrow,upper_f_limit,f,grid_size)

        cur_plot = cur_plot + grid_size;

    case 2

        [axes_array(cur_plot) width_subplot_spectrogram] = draw_balance (num_graphs, cur_plot,spec_y,w_size_bal,o_size_bal,upper_f_limit,f,grid_size)

        cur_plot = cur_plot + grid_size;

    case 3


        [axes_array(cur_plot) width_subplot_spectrogram]  = draw_wide (num_graphs, cur_plot,spec_y,w_size_wide,o_size_wide,upper_f_limit,f,grid_size)

        cur_plot = cur_plot + grid_size;

    case 4

        axes_array(cur_plot) = draw_best_chirp (num_graphs, cur_plot, values,array, results,freqs,width_subplot_spectrogram,deck_param,f,log_en,grid_size)

        cur_plot = cur_plot + grid_size;

    case 5

        axes_array(cur_plot) = draw_chirp_id(num_graphs, cur_plot,values, indices,results,width_subplot_spectrogram, chirp_type_key,f, deck_param, freqs,cmap,color_grid_a,x_axis_a,y_axis_cm,grid_size,draw_parameters)

        cur_plot = cur_plot + grid_size;
        
	case 6

        axes_array(cur_plot) = draw_best_chirp_of_amptype(num_graphs, cur_plot, values,array, results,freqs,width_subplot_spectrogram,deck_param,f,log_en, 1, chirp_type_key,grid_size)

        cur_plot = cur_plot + grid_size;
        
	case 7

        axes_array(cur_plot) = draw_best_chirp_of_amptype(num_graphs, cur_plot, values,array, results,freqs,width_subplot_spectrogram,deck_param,f,log_en, 2, chirp_type_key,grid_size)

        cur_plot = cur_plot + grid_size;

  	case 8

        axes_array(cur_plot) = draw_best_chirp_of_amptype(num_graphs, cur_plot, values,array, results,freqs,width_subplot_spectrogram,deck_param,f,log_en, 3, chirp_type_key,grid_size)

        cur_plot = cur_plot + grid_size;
        
	case 9

        axes_array(cur_plot) = draw_chirp_id_amp_type(num_graphs, cur_plot,values, indices,results,width_subplot_spectrogram, chirp_type_key,f, deck_param ,freqs,1,cmap ,grid_size,color_grid_a,x_axis_a,y_axis_cm,draw_parameters)
        cur_plot = cur_plot + grid_size;
        
	case 10

        axes_array(cur_plot) = draw_chirp_id_amp_type(num_graphs, cur_plot,values, indices,results,width_subplot_spectrogram, chirp_type_key,f, deck_param ,freqs,2,cmap,grid_size,color_grid_a,x_axis_a,y_axis_cm,draw_parameters)

        cur_plot = cur_plot + grid_size;
        
	case 11

        axes_array(cur_plot) = draw_chirp_id_amp_type(num_graphs, cur_plot,values, indices,results,width_subplot_spectrogram, chirp_type_key,f, deck_param ,freqs,3,cmap,grid_size,color_grid_a,x_axis_a,y_axis_cm,draw_parameters)

        cur_plot = cur_plot + grid_size;


    case 12

        
        
        best_slope_v_time(num_graphs, cur_plot,all_desc,results, deck_param, grid_size);

        cur_plot = cur_plot + grid_size;
        
        
	case 13

        axes_array(cur_plot) = draw_chirp_id_amp_type(num_graphs, cur_plot,values, indices,results,width_subplot_spectrogram, chirp_type_key,f, deck_param ,freqs,[2 3],cmap,grid_size,color_grid_a,x_axis_a,y_axis_cm,draw_parameters)

        cur_plot = cur_plot + grid_size;
        
	case 14

        axes_array(cur_plot) = draw_best_chirp_of_amptype(num_graphs, cur_plot, values,array, results,freqs,width_subplot_spectrogram,deck_param,f,log_en, [2 3], chirp_type_key,grid_size)

        cur_plot = cur_plot + grid_size;

    end
end





linkaxes(axes_array,'xy')




%Look at the distribution of matches to the signal above threshold.
% 
% figure;
% temp = reshape(indices,[], 1);
% temp(temp == 0) = [];
% histogram(temp)





function [ax width_subplot_spectrogram] = draw_narrow(total_num, sub_num,spec_y,w_size_narrow,o_size_narrow,upper_f_limit,f,grid_size)



    ax =  subplot(total_num,grid_size,[sub_num:1:sub_num+grid_size-2])



    [s,f_,t] = spectrogram(spec_y,w_size_narrow,o_size_narrow,1024,f,'yaxis');


    %Lop off the top 5 kHz of the spectrogram. 
    temp = find(f_ > upper_f_limit);
    f_(temp(1):end) = [];
    s(temp(1):end,:) = [];


    s = 10*log10(abs(s)+eps);
    s = s - max(max(s));

    f_ = f_/1000;

    hndl = imagesc(t, f_, s);
    hndl.Parent.YDir = 'normal';
    xlabel('(s)')
    ylabel('Frequency (kHz)')
    cblabel = 'dB'
    h = colorbar;
    h.Label.String = cblabel;

    temp = get(ax,'Position');
    width_subplot_spectrogram = temp(3);

    h = gcf;
    title('Narrowband Spectrogram')
%     set(findall(gcf,'-property','FontSize'),'FontSize',18)



end



function [ax width_subplot_spectrogram] = draw_wide(total_num, sub_num,spec_y,w_size_wide,o_size_wide,upper_f_limit,f,grid_size)



    ax =  subplot(total_num,grid_size,[sub_num:1:sub_num+grid_size-2])




    [s,f_,t] = spectrogram(spec_y,w_size_wide,o_size_wide,1024,f,'yaxis');

    %Lop off the top 5 kHz of the spectrogram. 
    temp = find(f_ > upper_f_limit);
    f_(temp(1):end) = [];
    s(temp(1):end,:) = [];

    f_ = f_/1000;

    s = 10*log10(abs(s)+eps);
    s = s - max(max(s));

    hndl = imagesc(t, f_, s);
    hndl.Parent.YDir = 'normal';
    xlabel('(s)')
    ylabel('Frequency (kHz)')
    cblabel = 'dB'
    h = colorbar;
    h.Label.String = cblabel;

    temp = get(ax,'Position');
    width_subplot_spectrogram = temp(3);


    h = gcf;
    title('Wideband Spectrogram')
%     set(findall(gcf,'-property','FontSize'),'FontSize',18)


end


function [ax width_subplot_spectrogram] = draw_balance(total_num, sub_num,spec_y,w_size_bal,o_size_bal,upper_f_limit,f,grid_size)


    ax =  subplot(total_num,grid_size,[sub_num:1:sub_num+grid_size-2])



    [s,f_,t] = spectrogram(spec_y,w_size_bal,o_size_bal,1024,f,'yaxis');

    %Lop off the top 5 kHz of the spectrogram. 
    temp = find(f_ > upper_f_limit);
    f_(temp(1):end) = [];
    s(temp(1):end,:) = [];

    s = 10*log10(abs(s)+eps);
    s = s - max(max(s));

    f_ = f_/1000;

    hndl = imagesc(t, f_, s);
    hndl.Parent.YDir = 'normal';
    xlabel('(s)')
    ylabel('Frequency (kHz)')
    cblabel = 'dB'
    h = colorbar;
    h.Label.String = cblabel;


    %Grab width of the spectrogram subplot so I can match it with other plots. 
    temp = get(ax,'Position');
    width_subplot_spectrogram = temp(3);

    h = gcf;
    title('Balanced Spectrogram')
%     set(findall(gcf,'-property','FontSize'),'FontSize',18)

end


function [ax] = draw_best_chirp(total_num, sub_num, values,array, results,freqs,width_subplot_spectrogram, deck_param,f, log_en,grid_size)
%We use the indices array to color codes out the types. 



    ax =  subplot(total_num,grid_size,[sub_num:1:sub_num+grid_size-2])

    
	sample_step = deck_param.time_sample_step;
    
    
    

    array = max(array);
    array = squeeze(array);
    x_axis = [0:sample_step:length(results(:,1))*sample_step-1].*((1/f));
    y_axis = freqs / 1000;
    
    %Old way of doing it. Imagesc doesn't handle nonlinear axis (foveation)
    %nicely. Changed to pcolor. 
    imagesc(x_axis,y_axis,(array))
    set(gca,'YDir','normal')
    
    h = pcolor(x_axis,y_axis,array);
    h.EdgeColor = 'none';
    
    
    
    h = colorbar
    if  log_en == 1
    %ylabel(h, '20*log_{10}(Norm Dot Result)')
    ylabel(h, 'dB')
    else
    ylabel(h, 'Norm Dot Result [0 to 1]')  
    end
    title('Best Chirplet of Deck Match')
    xlabel('(s)')
    ylabel('Frequency (kHz)')
% 	set(findall(gcf,'-property','FontSize'),'FontSize',18)
    
    

    

end


function [ax] = draw_best_chirp_of_amptype(total_num, sub_num, values,array, results,freqs,width_subplot_spectrogram, deck_param,f, log_en,type, chirp_type_key,grid_size)
%We use the indices array to color codes out the types. 


    %The full array has all chirp types. 
    %If we want to grab only a certain type of chirp (guass,up,down) we
    %index into full array and extract all of type. 
  
    type_indices = [];
    indices_temp = [];
    array_copy = 0.* array;
	new_array = zeros(length(array(1,:,1)),length(array(1,1,:)));
     for k_1 = 1:length(type)
    
       indices_temp = find(chirp_type_key(4,:) == type(k_1));
       
       type_indices(k_1,:) = indices_temp;

     end
     
	%Take the max across those types. 
    %Keep the indice spots for extraction later. 

     
     if (length(type) > 1)

 
        array_copy(type_indices(1,:,:),:,:) = array(type_indices(1,:),:,:);
        array_copy(type_indices(2,:,:),:,:) = array(type_indices(2,:),:,:);
        [M I] = max(array_copy);
         
        %temp = array(type_indices(2,:),:,:) + max(reshape(array(type_indices(1,:),:),1,[]));
        %Offset all the second type by 1. 

        temp = array(type_indices(1,:),:,:) - 1; 
        
        %These define ranges of new colormap. 

         %Put the offset values back into the array_copy;
         %array_copy(type_indices(1,:,:),:,:) = array(type_indices(1,:),:,:);
         array_copy(type_indices(1,:,:),:,:) = temp;
         
         
        
         

         for i = 1:length(array_copy(1,:,1));
             for o = 1:length(array_copy(1,1,:));

                 new_array(i,o) = array_copy(I(1,i,o),i,o);


             end
         end

         
         
         
     else 
        new_array = array(reshape(type_indices,1,[]),:,:); 
    
		new_array = max(new_array);
     end



    array = squeeze(new_array);



    
    type_string = [];

    for k = 1:length(type)
        type_string = [type_string ' ' deck_param.type_strings{type(k)}];
    end



    ax =  subplot(total_num,grid_size,[sub_num:1:sub_num+grid_size-2])

    
	sample_step = deck_param.time_sample_step;
    
    
    x_axis = [0:sample_step:length(results(:,1))*sample_step-1].*((1/f));
    y_axis = freqs / 1000;
    
    
    imagesc(x_axis,y_axis,(array))
    set(gca,'YDir','normal')
    h = colorbar
    
    if  log_en == 1
    %ylabel(h, '20*log_{10}(Norm Dot Result)')
    ylabel(h, 'dB')
    else
    ylabel(h, 'Norm Dot Result [0 to 1]')  
    end
    title(['Best ' type_string ' of Deck Match'] )
    xlabel('(s)')
    ylabel('Frequency (kHz)')

end


function [ax] = draw_chirp_id(total_num, sub_num,values, indices,results,width_subplot_spectrogram, chirp_type_key,f, deck_param, freqs,cmap,color_grid_a,x_axis_a,y_axis_cm,grid_size,draw_parameters)


    ax =  subplot(total_num,grid_size,[sub_num:1:sub_num+grid_size-2])

    
	sample_step = deck_param.time_sample_step;
 	x_axis = [0:sample_step:length(results(:,1))*sample_step-1].*((1/f));
    y_axis = freqs / 1000;
    

    
    

    
    if (draw_parameters.scatter_en == 0)
     	hdl = imagesc(x_axis,y_axis,indices)  
        
        h = pcolor(x_axis,y_axis,indices);
        h.EdgeColor = 'none';
        
        
        
        
    else  
            
    %Alternatively plot the indices as points and force x/y axis.
        [a b] = find(indices > 0);
        %Remap the x and y axis like so. 
        X = sub2ind(size(indices), a, b);
        scatter3(x_axis(b),y_axis(a),indices(X),75,indices(X),'filled')
    end



    %Apply a transparency map to the imagesc(). 
    %Values contains the related values for each index. 
    %Scale that as a transparency mask and apply it. 

    %Values is already! 0-1.0 based on dot product match.
    %So it is already an acceptable alpha mask. 
    if draw_parameters.transparency_en == 1
        
        alphamask = squeeze(values);
        

        set(hdl, 'AlphaData', alphamask);
    end






    
    

    set(gca,'YDir','normal')
    title('Chirplet Best Match Index')
    xlabel('(s)')
    ylabel('Frequency (kHz)')
%     set(findall(gcf,'-property','FontSize'),'FontSize',18)




    temp = get(ax,'Position');
    temp(3) = width_subplot_spectrogram;
    set(ax,'Position',temp);



    %Add a bottom color to represent below threshold. 
    cmapn = [[255 255 255]/255; cmap];
    colormap(ax,cmapn);
    %Use climit to get the mapping we want. 
    %0 - num_colors/number of chirplet types. 
    %Where are the current limits?
    c1 = caxis;
    caxis([0 length(cmap)])
    
	view(0,90);


    %Inset the chirplet into the plot
    % Place second set of axes on same plot
    %Draw all chirp types. 

    
%     temp = get(ax,'Position');
%     %[left bottom width height]
%     temp_new = temp;
%     temp_new(1) = temp(1)+ temp(3);
%     temp_new(2) = temp(2);
%     temp_new(3) = .3;
% 	temp_new(4) = .3;
%     
%     handaxes2 = axes('Position', temp_new);
    
    graphic_type = 1;
    
    if (graphic_type == 1)

        ax2 =  subplot(total_num,grid_size,sub_num+grid_size-1)
        
        image(x_axis_a,y_axis_cm,color_grid_a);
        colormap(ax2,cmap);
        %Use climit to get the mapping we want. 
        %0 - num_colors/number of chirplet types. 
        %Where are the current limits?
        c1 = caxis;
        caxis([1 length(cmap)])
        
        xticks(x_axis_a);
        yticks([y_axis_cm(1) y_axis_cm(end)]);
        
        xlabel('Amp Types');
        ylabel('Chirp Slopes');
        
    else

        %Draw all amp types or a single amp type.
        draw_type = 0

    	draw_chirplet_stack(deck_param,chirp_type_key,draw_type,cmap);
       	axis off;
    end
    




    %Draw the chirplet deck next to the subplot. 

end

function [ax] = draw_chirp_id_amp_type(total_num, sub_num,values, indices,results,width_subplot_spectrogram, chirp_type_key,f, deck_param, freqs, type, cmap,grid_size,color_grid_a,x_axis_a,y_axis_cm,draw_parameters)


    %Here we show ID of only! the chirps matching a certain amplitude profile.
    %The max operator has already been taken. Thus we show only chirp ID's
    %or a certain type which are best. 
    
    %Idea here is to show that best chirps are of a certain profile in
    %certain spots of the signal (onset/offset/middle)

    ax =  subplot(total_num,grid_size,[sub_num:1:sub_num+grid_size-2])

    
	sample_step = deck_param.time_sample_step;
 	x_axis = [0:sample_step:length(results(:,1))*sample_step-1].*((1/f));
    y_axis = freqs / 1000;
    

    
    %Supress the indices we don't want to plot.
    
    indices_copy = indices;
    indices_temp = zeros(length(indices(:,1)),length(indices(1,:)));
    for k_1 = 1:length(type)
    
        type_indices = find(chirp_type_key(4,:) == type(k_1));

        %Find the indices of interest. 
        running_a = [];
        running_b = [];
        for k = 1:length(type_indices)

            [a b] = find(indices ==  type_indices(k)); 
            running_a = [running_a a'];
            running_b = [running_b b'];
        end

        supress_i = zeros(length(indices(:,1)),length(indices(1,:)));


        X = sub2ind(size(supress_i), running_a, running_b);
        supress_i(X) = 1;

        %Create a 0/1's matrix where we will zero out idents we don't want. 
        
        
        indices_temp = indices_temp + (indices_copy .* supress_i);
    
    end
    
    
    indices = indices_temp;
  
    
    
    
    
    
    
    

    if (draw_parameters.scatter_en == 0)
     	hdl = imagesc(x_axis,y_axis,indices)  
    else  
            
    %Alternatively plot the indices as points and force x/y axis.
        [a b] = find(indices > 0);
        
        X = sub2ind(size(indices), a, b);
        %Remap the x and y axis like so. 
        scatter3(x_axis(b),y_axis(a),indices(X),75,indices(X),'filled')
    end
    
    
	 if draw_parameters.transparency_en == 1
        
        alphamask = squeeze(values);
        

        set(hdl, 'AlphaData', alphamask);
    end
    
    
    

    set(gca,'YDir','normal')
    %title(['Chirplet Best Match of Amp Type ' string(type_strings{type}) ' Index'])
    xlabel('(s)')
    ylabel('Frequency (kHz)')
%     set(findall(gcf,'-property','FontSize'),'FontSize',18)




    temp = get(ax,'Position');
    temp(3) = width_subplot_spectrogram;
    set(ax,'Position',temp);



    %Add a bottom color to represent below threshold. 
    cmapn = [[255 255 255]/255; cmap];
    colormap(ax,cmapn);
    %Use climit to get the mapping we want. 
    %0 - num_colors/number of chirplet types. 
    %Where are the current limits?
    c1 = caxis;
    caxis([0 length(cmap)])
    
    view(0,90);


    %Inset the chirplet into the plot
    % Place second set of axes on same plot
	temp = get(ax,'Position');
    %[left bottom width height]
    temp_new = temp;
    temp_new(1) = temp(1)+ temp(3);
    temp_new(2) = temp(2);
    temp_new(3) = .2;
	temp_new(4) = .2;
    
    handaxes2 = axes('Position', temp_new);
    %Draw only the amp type.


 graphic_type = 1;
    
    if (graphic_type == 1)

        ax2 =  subplot(total_num,grid_size,sub_num+grid_size-1)
        
        image(x_axis_a(type),y_axis_cm,color_grid_a(:,type));
        colormap(ax2,cmap);
        %Use climit to get the mapping we want. 
        %0 - num_colors/number of chirplet types. 
        %Where are the current limits?
        c1 = caxis;
        caxis([1 length(cmap)])
        xticks(x_axis_a);
        yticks([y_axis_cm(1) y_axis_cm(end)]);
        xlabel('Amp Types');
        ylabel('Chirp Slopes');
        
    else

        %Draw all amp types or a single amp type.
     
        draw_type = type;
        draw_chirplet_stack(deck_param,chirp_type_key,draw_type,cmap);
        axis off;
    end









    %Draw the chirplet deck next to the subplot. 

end



function [ax] = best_slope_v_time(total_num, sub_num,all_desc,results, deck_param, grid_size)

    ax =  subplot(total_num,grid_size,[sub_num:1:sub_num+grid_size-2])

    %Extract out only the guassin types. 
    guass_rows = find(all_desc(:,9) == 1);
    
    

   	%Mind the max value of each column of results 
    max_result = [];

    for k = 1:length(results(:,1))


    [max_v index] = sort(results(k,:),'descend');
    k1 = 1;
        while (~ismember(index(k1),guass_rows))
            k1 = k1+1;
        end
        
           	max_result(k,1) = index(k1);
            max_result(k,2) = max_v(k1);
    end
    
   

    
	sample_step = deck_param.time_sample_step;
    f = deck_param.f;
 	x_axis = [0:sample_step:length(results(:,1))*sample_step-1].*((1/f));
    
    slopes = [];
    %Find the best slope match. 
    for i = 1:length(max_result)

                type = max_result(i);
             	slopes(i) = all_desc(type,3);
        
    end
    
    plot(x_axis,slopes)
    xlabel('(s)')
    ylabel('Slope of Best Chirp Match at TF Point')
    

end

function [ax] = draw_best_chirp_of_slopetype(total_num, sub_num, values,array, results,freqs,width_subplot_spectrogram, deck_param,f, log_en, chirp_type_key,grid_size,draw_parameters)
%We use the indices array to color codes out the types. 


    %The full array has all chirp types. 
    %If we want to grab only a certain type of chirp (guass,up,down) we
    %index into full array and extract all of type. 
  
   
    indices_temp = [];
    array_copy = 0.* array;

	%Take the max across those types. 
    %Keep the indice spots for extraction later. 

     
	array_copy(type_indices(type_indices,:,:),:,:) = array(type_indices,:,:);



	new_array = max(new_array);
    array = squeeze(new_array);



    
    type_string = [];

    for k = 1:length(type)
        type_string = [type_string ' ' deck_param.type_strings{type(k)}];
    end



    ax =  subplot(total_num,grid_size,[sub_num:1:sub_num+grid_size-2])

    
	sample_step = deck_param.time_sample_step;
    
    
    x_axis = [0:sample_step:length(results(:,1))*sample_step-1].*((1/f));
    y_axis = freqs / 1000;
    
    
    imagesc(x_axis,y_axis,(array))
    set(gca,'YDir','normal')
    h = colorbar
    
    if  log_en == 1
    %ylabel(h, '20*log_{10}(Norm Dot Result)')
    ylabel(h, 'dB')
    else
    ylabel(h, 'Norm Dot Result [0 to 1]')  
    end
    title(['Best ' type_string ' of Deck Match'] )
    xlabel('(s)')
    ylabel('Frequency (kHz)')

end
