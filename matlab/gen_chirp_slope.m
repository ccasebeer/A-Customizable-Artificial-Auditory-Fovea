function [chirplet start stop slope_out] = gen_chirp_slope(center_f,slope,t,f,type)



        k = length(t);
        
        add = (slope*k*(1/(f)))/2;
        %add = slope/2;
        
        stop = center_f + add;
        start = center_f - add;
        
        
        
        %Maintain slopes rather than cutting them off. 
        %If you have to increase start or stop to maintain a
        %slope, do it. 
        
        if(start < 0)
            
            dif = abs(start);
            stop = stop + dif;
            
            start = 0;
        end
        
        if(stop < 0)
            
            dif = abs(stop);
            start = start + dif;
            stop = 0;
            
        end
        
        
        %Y = chirp(T,F0,T1,F1)
        
        chirpy = chirp(t,start,t(end),stop);
        
        if (strcmp(type,'gauss'))
            
            let = gausswin(length(chirpy))';
           
        elseif (strcmp(type,'up_lin'))
                
            let = linspace(0,1,length(chirpy'));     
        
        elseif (strcmp(type,'down_lin')) 
                
            let = linspace(1,0,length(chirpy'));
        
        else
           return 
        end
        
        

        
        
        slope_out  = (stop-start)/(k/f);

        chirplet = chirpy .* let;



end