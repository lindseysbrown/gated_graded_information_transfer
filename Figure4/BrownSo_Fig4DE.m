function BrownSo_Fig4DE(coh_file, grouping_file)

% -------------------------------------------------------------------------
% BrownSo_Fig4DE
%
% Reproduces Figure 4DE on temporal dynamics of decision-related information in sender and receiver neurons
% D: Temporal dynamics with individual fits (gray) and population averages (blue/red) 
% E:  Histogram of fitted tau values
%
% Based on So & Shadlen (2022), extended for modeling analysis.
%
% INPUT:
%   coh_file : path to CohDep dataset (.mat) 
%   grouping_file : path to SenderReceiver grouping info (.mat)
%
% -------------------------------------------------------------------------

%% Load
data = load(coh_file);
grp  = load(grouping_file);

CohDep_30 = data.CohDep_30;
CohDep_31 = data.CohDep_31;
p_30 = data.p_30;
p_31 = data.p_31;

%% Params
steps = 50;
plot_t_length = 1000;
plot_t_bins = plot_t_length / steps;

p_crit = 0.05;
baseadjust = 1;

tau_tarray_fitPlot = 0:plot_t_length-steps;
tau_tarray_fit     = 0:steps:plot_t_length-steps;

exp_func = @(params, t) params(1) * (exp(-t / params(2)) + params(3));
initial_guess = [1, 200, 0.01];
options = optimset('Display','off');

%% Init storage 
senderno = 0;
receiverno = 0;

tau_plot_s = nan(1000, plot_t_bins);
tau_plot_r = nan(1000, plot_t_bins);

tau_fit_s = nan(1,1000);
tau_fit_r = nan(1,1000);

cellname_s = {};
cellname_r = {};

tau_tarray_fitPlot_s = {};
tau_tarray_fitPlot_r = {};

ktau_fitPlot_s_indiv = {};
ktau_fitPlot_r_indiv = {};

%% ===================== SENDERS =====================

process_sender_group(grp.RRpairSenderA, CohDep_30);
process_sender_group(grp.GGpairSenderA, CohDep_30);
process_sender_group(grp.RRpairSenderB, CohDep_31);
process_sender_group(grp.GGpairSenderB, CohDep_31);

%% ===================== RECEIVERS =====================

process_receiver_group(grp.RRpairReceiverA, CohDep_30);
process_receiver_group(grp.GGpairReceiverA, CohDep_30);
process_receiver_group(grp.RRpairReceiverB, CohDep_31);
process_receiver_group(grp.GGpairReceiverB, CohDep_31);

%% Trim
tau_plot_s = tau_plot_s(1:senderno,:);
tau_plot_r = tau_plot_r(1:receiverno,:);

tau_fit_s = tau_fit_s(1:senderno);
tau_fit_r = tau_fit_r(1:receiverno);

fprintf('Senders: %d | Receivers: %d\n', senderno, receiverno);

%% ===================== IQR CLEAN =====================
s_q1 = quantile(tau_fit_s,0.25); s_q3 = quantile(tau_fit_s,0.75);
s_lb = s_q1 - 1.5*(s_q3-s_q1); s_ub = s_q3 + 1.5*(s_q3-s_q1);
clean_ts_id = tau_fit_s>=s_lb & tau_fit_s<=s_ub;
clean_ts = tau_fit_s(clean_ts_id);

r_q1 = quantile(tau_fit_r,0.25); r_q3 = quantile(tau_fit_r,0.75);
r_lb = r_q1 - 1.5*(r_q3-r_q1); r_ub = r_q3 + 1.5*(r_q3-r_q1);
clean_tr_id = tau_fit_r>=r_lb & tau_fit_r<=r_ub;
clean_tr = tau_fit_r(clean_tr_id);

[~, p_ttest] = ttest2(clean_ts,clean_tr);
p_ranksum = ranksum(clean_ts,clean_tr);
[~, p_ks] = kstest2(clean_ts,clean_tr);

mean_ts = mean(clean_ts)
mean_tr = mean(clean_tr)
fprintf('W/o outliers: t-test p=%.3g ranksum p=%.3g ks test p=%.3g\n',p_ttest,p_ranksum,p_ks);

[~, p_ttest] = ttest2(tau_fit_s,tau_fit_r);
p_ranksum = ranksum(tau_fit_s,tau_fit_r);
[~, p_ks] = kstest2(tau_fit_s,tau_fit_r);

fprintf('Including outliers: t-test p=%.3g ranksum p=%.3g ks test p=%.3g\n',p_ttest,p_ranksum,p_ks);


%% ===================== HIST =====================
figure;
subplot(211); histogram(clean_ts,30,'FaceColor','b'); title('Sender'); box off; fig_setting();
subplot(212); histogram(clean_tr,30,'FaceColor','r'); title('Receiver'); box off; fig_setting();

%% ===================== GRAND =====================
plot_ktau_s = nanmean(tau_plot_s(clean_ts_id,:),1);
plot_ktau_r = nanmean(tau_plot_r(clean_tr_id,:),1);

plot_ktau_se_s = nanstd(tau_plot_s(clean_ts_id,:))./sqrt(sum(clean_ts_id));
plot_ktau_se_r = nanstd(tau_plot_r(clean_tr_id,:))./sqrt(sum(clean_tr_id));

params_s = lsqcurvefit(exp_func, initial_guess, tau_tarray_fit, plot_ktau_s, [], [], options);
params_r = lsqcurvefit(exp_func, initial_guess, tau_tarray_fit, plot_ktau_r, [], [], options);

ktau_fitPlot_s = exp_func(params_s, tau_tarray_fitPlot);
ktau_fitPlot_r = exp_func(params_r, tau_tarray_fitPlot);

%% ===================== MAIN FIG =====================
figure; hold on;

for i = 1:senderno
    if clean_ts_id(i)
        plot(tau_tarray_fitPlot_s{i}, ktau_fitPlot_s_indiv{i}, ...
            'Color',[0.7 0.7 0.7],'LineWidth',0.5);
    end
end

for i = 1:receiverno
    if clean_tr_id(i)
        plot(-tau_tarray_fitPlot_r{i}, ktau_fitPlot_r_indiv{i}, ...
            'Color',[0.7 0.7 0.7],'LineWidth',0.5);
    end
end

plot(tau_tarray_fitPlot, ktau_fitPlot_s,'b','LineWidth',2);
plot(-tau_tarray_fitPlot, ktau_fitPlot_r,'r','LineWidth',2);

errorbar(tau_tarray_fit, plot_ktau_s, plot_ktau_se_s,...
    'b','LineStyle','none','Marker','o','MarkerFaceColor','b');

errorbar(-tau_tarray_fit, plot_ktau_r, plot_ktau_se_r,...
    'r','LineStyle','none','Marker','o','MarkerFaceColor','r');

ylim([0 0.35]);
xlabel('Time (ms)');
ylabel('|Kendall \tau|');
hold off;
fig_setting();

%% ===================== FUNCTIONS =====================

    function process_sender_group(group, CohDep)
        filename_array = CohDep(:,9);

        for ii = 2:length(group)

            fname = group{ii,2};
            idx = find(strcmp(filename_array,fname),1);
            if isempty(idx), continue; end
            if isempty(CohDep{idx,2}), continue; end

            tau = abs(CohDep{idx,2+baseadjust*8});
            if any(isnan(tau)) || max(tau)==0, continue; end

            [~,peak] = max(tau);
            tau_dyn = tau(peak:end);
            t_dyn = steps*(0:length(tau_dyn)-1);

            if length(tau_dyn)<2, continue; end

            % ===== EXACT dedup logic =====
            if (senderno==0) || ~strcmp(cellname_s{senderno},fname)

                senderno = senderno+1;
                cellname_s{senderno} = fname;

                params = lsqcurvefit(exp_func,initial_guess,t_dyn,tau_dyn,[],[],options);

                tau_fit_s(senderno) = params(2);

                col = min(length(tau_dyn),plot_t_bins);
                tau_plot_s(senderno,1:col) = tau_dyn(1:col);

                tau_tarray_fitPlot_s{senderno} = t_dyn(1:col);
                ktau_fitPlot_s_indiv{senderno} = exp_func(params, tau_tarray_fitPlot_s{senderno});

            end
        end
    end

    function process_receiver_group(group, CohDep)
        filename_array = CohDep(:,9);

        for ii = 2:length(group)

            fname = group{ii,2};
            idx = find(strcmp(filename_array,fname),1);
            if isempty(idx), continue; end
            if isempty(CohDep{idx,2}), continue; end

            tau = abs(CohDep{idx,5+baseadjust*8});
            if any(isnan(tau)) || max(tau)==0, continue; end

            [~,peak] = max(tau);

            tau_dyn = tau(1:peak);
            tau_dyn = fliplr(tau_dyn);
            t_dyn = steps*(0:length(tau_dyn)-1);

            if length(tau_dyn)<2, continue; end

            % ===== EXACT dedup logic =====
            if (receiverno==0) || ~strcmp(cellname_r{receiverno},fname)

                receiverno = receiverno+1;
                cellname_r{receiverno} = fname;

                params = lsqcurvefit(exp_func,initial_guess,t_dyn,tau_dyn,[],[],options);

                tau_fit_r(receiverno) = params(2);

                col = min(length(tau_dyn),plot_t_bins);
                tau_plot_r(receiverno,end-col+1:end) = tau_dyn(1:col);

                tau_tarray_fitPlot_r{receiverno} = t_dyn(1:col);
                ktau_fitPlot_r_indiv{receiverno} = exp_func(params, tau_tarray_fitPlot_r{receiverno});

            end
        end
    end

end