% Function to get and display time
% Sathwik Chadaga, Nov 3, 2021
function timeValue = displayTime(message, timeValue)
    if(nargin<2)
        timeValue = clock;
    end
    fprintf([message num2str(timeValue(4)) ':' num2str(timeValue(5)) ':' num2str(timeValue(6)) ' (HH:MM:SS.ssss)\n'])
end