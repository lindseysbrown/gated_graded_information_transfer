function BrownSo_Fig4AB(data_file)
% -------------------------------------------------------------------------
% BrownSo_Fig4AB
%
% Reproduces Figure 4AB: 
% A: Identification of "midway" neurons and heatmaps,
% B: A bar graph comparing peak information at different epochs for each midway neuron
%
% Midway neurons are defined as neurons whose decision-related signal
% (|Kendall tau|) peaks during pursuit (200-400 ms after pursuit onset)
% and exceeds a threshold (tau > 0.05).
%
% Based on So & Shadlen (2022), extended for modeling analysis.
%
% INPUT:
%   data_file : path to CohDep dataset (.mat)
%
% -------------------------------------------------------------------------

%% Load data
data = load(data_file);

CohDep_30 = data.CohDep_30;
CohDep_31 = data.CohDep_31;
p_30 = data.p_30;
p_31 = data.p_31;
TimeArray = data.TimeArray;

% CohDep structure:
%   {i,10} = P1-aligned Kendall tau
%   {i,13} = saccade-aligned Kendall tau
%   {i,14} = pursuit-aligned Kendall tau

%% Parameters
bin_size = 50;                 % ms
kt_crit = 0.05;               % Kendall tau threshold

% Time windows
pre_p1 = 0; post_p1 = 500;
pre_sacc = -300; post_sacc = 300;
pre_pursuit = -400; post_pursuit = 800;

% Peak windows
pre_p1_peak = 100; post_p1_peak = 400;
pre_sacc_peak = -150; post_sacc_peak = 150;
pre_pursuit_peak = 150; post_pursuit_peak = 450;

% Midway criterion window (relative to pursuit onset)
crit_start = 200;
crit_end   = 400;

caxis_range = [-0.15 0.15];

%% Indexing
pre_p1_idx = find(TimeArray{1,2} == pre_p1);
post_p1_idx = find(TimeArray{1,2} == post_p1);

pre_sacc_idx = find(TimeArray{2,2} == pre_sacc);
post_sacc_idx = find(TimeArray{2,2} == post_sacc);

pre_pursuit_idx = find(TimeArray{3,2} == pre_pursuit);
post_pursuit_idx = find(TimeArray{3,2} == post_pursuit);

pursuit_array = pre_pursuit:bin_size:post_pursuit;
midway_start_idx = find(pursuit_array == crit_start);
midway_end_idx   = find(pursuit_array == crit_end);

% Peak indices
pre_p1_peak_idx = find(TimeArray{1,2} == pre_p1_peak);
post_p1_peak_idx = find(TimeArray{1,2} == post_p1_peak);

pre_sacc_peak_idx = find(TimeArray{2,2} == pre_sacc_peak);
post_sacc_peak_idx = find(TimeArray{2,2} == post_sacc_peak);

pre_pursuit_peak_idx = find(TimeArray{3,2} == pre_pursuit_peak);
post_pursuit_peak_idx = find(TimeArray{3,2} == post_pursuit_peak);

%% Preallocate (maximum possible size)
max_cells = length(CohDep_30) + length(CohDep_31);

info_p1 = nan(max_cells, length(pre_p1:bin_size:post_p1));
info_sacc = nan(max_cells, length(pre_sacc:bin_size:post_sacc));
info_pursuit = nan(max_cells, length(pre_pursuit:bin_size:post_pursuit));

info_p1_peak = nan(max_cells,1);
info_sacc_peak = nan(max_cells,1);
info_pursuit_peak = nan(max_cells,1);

cell_count = 0;

%% Helper function for processing one dataset
    function process_dataset(CohDep, p_data)
        for i = 2:length(CohDep)

            if ~isempty(p_data{i,10})

                info_pursuit_array = abs(CohDep{i,14}(pre_pursuit_idx:post_pursuit_idx));
                [max_info, max_t] = max(info_pursuit_array);

                % --- Midway neuron criterion ---
                if (max_info > kt_crit) && ...
                   (max_t >= midway_start_idx) && ...
                   (max_t <= midway_end_idx)

                    cell_count = cell_count + 1;

                    info_p1(cell_count,:) = abs(CohDep{i,10}(pre_p1_idx:post_p1_idx));
                    info_sacc(cell_count,:) = abs(CohDep{i,13}(pre_sacc_idx:post_sacc_idx));
                    info_pursuit(cell_count,:) = info_pursuit_array;

                    % Peak values
                    info_pursuit_peak(cell_count) = max_info;

                    temp_p1 = abs(CohDep{i,10}(pre_p1_peak_idx:post_p1_peak_idx));
                    info_p1_peak(cell_count) = max(temp_p1);

                    temp_sacc = abs(CohDep{i,13}(pre_sacc_peak_idx:post_sacc_peak_idx));
                    info_sacc_peak(cell_count) = max(temp_sacc);

                end
            end
        end
    end

%% Process datasets
process_dataset(CohDep_30, p_30);
process_dataset(CohDep_31, p_31);

fprintf('Total midway neurons: %d\n', cell_count);

%% Trim unused preallocated rows
info_p1 = info_p1(1:cell_count,:);
info_sacc = info_sacc(1:cell_count,:);
info_pursuit = info_pursuit(1:cell_count,:);

info_p1_peak = info_p1_peak(1:cell_count);
info_sacc_peak = info_sacc_peak(1:cell_count);
info_pursuit_peak = info_pursuit_peak(1:cell_count);

%% Sort neurons by peak timing during pursuit
[~, peak_idx] = max(info_pursuit, [], 2);
[~, sort_order] = sort(peak_idx);

info_p1_sorted = info_p1(sort_order,:);
info_sacc_sorted = info_sacc(sort_order,:);
info_pursuit_sorted = info_pursuit(sort_order,:);

%% Heatmaps
figure;

subplot(1,4,1)
imagesc(info_p1_sorted);
xticks([1 3 5 7 9]);
xticklabels({'0', '100', '200', '300', '400', '500'}); 
hold on; xline(0, 'w--'); hold off;
title('P1-aligned');
ylabel('Neuron (sorted)');
caxis(caxis_range);
%colorbar;

subplot(1,4,2)
imagesc(info_sacc_sorted);
xticks([3 5 7 9 11]);
xticklabels({'-200', '-100', '0', '100', '200'}); 
hold on; xline(0, 'w--'); hold off;
title('Saccade-aligned');
caxis(caxis_range);

subplot(1,4,[3 4])
imagesc(info_pursuit_sorted);
xticks([1 3 5 7 9 11 13 15 17 19 21 23]);
xticklabels({'-400', '-300', '-200', '-100', '0', '100', '200', '300', '400', '500', '600', '700'}); 
hold on; xline(0, 'w--'); hold off;
title('Pursuit-aligned');
caxis(caxis_range);

%% Peak difference statistics
peak_diff1 = info_pursuit_peak - info_sacc_peak;
peak_diff2 = info_pursuit_peak - info_p1_peak;
peak_diff3 = info_sacc_peak - info_p1_peak;

[~, p_diff1] = ttest(peak_diff1);
[~, p_diff2] = ttest(peak_diff2);
[~, p_diff3] = ttest(peak_diff3);

fprintf('Pursuit - Sacc: p = %.3g\n', p_diff1);
fprintf('Pursuit - P1:   p = %.3g\n', p_diff2);
fprintf('Sacc - P1:      p = %.3g\n', p_diff3);

%% Bar plot
figure;
data = [mean(peak_diff1), mean(peak_diff2), mean(peak_diff3)];
errors = [std(peak_diff1), std(peak_diff2), std(peak_diff3)] ./ sqrt(cell_count);

bar(1:3, data); hold on;
errorbar(1:3, data, errors, 'k', 'linestyle','none');

xticklabels({'Pursuit - Sacc','Pursuit - P1','Sacc - P1'});
ylabel('Difference in peak info');
title('Peak information differences');

end