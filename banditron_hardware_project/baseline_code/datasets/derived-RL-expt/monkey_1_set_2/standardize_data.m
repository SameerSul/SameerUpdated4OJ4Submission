input_dir = 'C:\Users\Sameer\Desktop\4OJ4 Stuff\RL-Algorithms-for-iBMI-Applications\banditron_hardware_project\baseline_code\datasets\original\';
output_dir = fullfile(input_dir, 'python_ready_data');
if ~exist(output_dir, 'dir'), mkdir(output_dir); end

raw_files = dir(fullfile(input_dir, 'IMETrainingData_*.mat'));

for i = 1:length(raw_files)
    try
        data = load(fullfile(input_dir, raw_files(i).name));
        
        % Check if the structure exists
        if isfield(data, 'IMETrainingData')
            % Logic from your successful clean_monkey_data.mat
            X = cell2mat(data.IMETrainingData.SentSignals);
            Y = cell2mat(data.IMETrainingData.SentCommand);
            
            % If Y is empty or all NaN, let's try a different field
            if all(isnan(Y)) || isempty(Y)
                % Fallback: some files use 'TargetID' or 'CursorCommand'
                % Check your MATLAB workspace for the correct label field
                fprintf('Warning: %s has no valid labels in SentCommand\n', raw_files(i).name);
            end

            save_name = fullfile(output_dir, ['clean_', raw_files(i).name]);
            save(save_name, 'X', 'Y', '-v7');
            fprintf('Processed: %s (Samples: %d)\n', raw_files(i).name, length(Y));
        end
    catch E
        fprintf('Error in %s: %s\n', raw_files(i).name, E.message);
    end
end