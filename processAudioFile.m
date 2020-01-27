%% takes in the title of mp3 file, start time (seconds) of when to sample, and the number
%% of num_samples to take as parameters and processes audio file into spectrogram.
function processAudioFile(title, startsample, num_samples)
    audio = strcat(title, '.mp3');
    audio_info = audioinfo(audio);
    
    Fs = audio_info.SampleRate;
    duration = 5; % sample for 5 secs
    d_sample = duration*Fs; % duration of sample
    
    if (startsample == 0) % for sampling at different times
        start = 1;
    else
        start = startsample+1;
    end
    
    count = 1;
    for begin_time = start:d_sample:num_samples*d_sample+startsample
        end_time = min(begin_time+d_sample-1, num_samples*d_sample+startsample);
        y = audioread(audio, [begin_time,end_time]);
        spectrogram(y(:,1), 1000, 500, 1000, Fs, 'yaxis');
        colorbar('off');
        set(gca, 'position',[0 0 1 1],'units','normalized')
        axis off
        fileName = strcat(title,sprintf('_%d.png', count));
        saveas(gcf, fileName)
        count = count+1;
    end
end