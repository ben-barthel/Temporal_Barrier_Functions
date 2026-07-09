function plot_paper_figures()
%PLOT_PAPER_FIGURES  Generate the paper's statistical figures + summary
%   tables from the saved experiment cells.
%
%   Prerequisites: run reproduce_random_sphere_experiments and/or
%   reproduce_formation_experiments first (their .mat outputs must exist
%   under ../Results/paper_random_sphere and ../Results/paper_formation).
%
%   Outputs (PNG + PDF + FIG each) are written to
%   ../Results/paper_figures/{random_sphere, formation}/, and the summary
%   statistics tables are printed to the console.

addpath(fullfile(fileparts(mfilename('fullpath')), '..'));
root = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'Results');

%% ---------------- Random-sphere figures (9 cells) ----------------
rs_dir  = fullfile(root, 'paper_random_sphere');
rs_save = fullfile(root, 'paper_figures', 'random_sphere');
nn_sfx  = 'attc_vraw__iw_huber3__data1.mat';

rs_cells = { ...
    struct('file', 'random_sphere__none.mat',                       'label', 'no CBF\newline0 pur'); ...
    struct('file', 'random_sphere_pursuit_adv0p9__none.mat',        'label', 'no CBF\newlineslow pur'); ...
    struct('file', 'random_sphere_pursuit_adv1p5__none.mat',        'label', 'no CBF\newlinefast pur'); ...
    struct('file', 'random_sphere__hocbf.mat',                      'label', 'HOCBF\newline0 pur'); ...
    struct('file', 'random_sphere_pursuit_adv0p9__hocbf.mat',       'label', 'HOCBF\newlineslow pur'); ...
    struct('file', 'random_sphere_pursuit_adv1p5__hocbf.mat',       'label', 'HOCBF\newlinefast pur'); ...
    struct('file', ['random_sphere__'                    nn_sfx],   'label', 'aTTC-CBF\newline0 pur'); ...
    struct('file', ['random_sphere_pursuit_adv0p9__'     nn_sfx],   'label', 'aTTC-CBF\newlineslow pur'); ...
    struct('file', ['random_sphere_pursuit_adv1p5__'     nn_sfx],   'label', 'aTTC-CBF\newlinefast pur')};

[Ss, labels, ok] = load_cells(rs_dir, rs_cells);
if ok
    fprintf('\n=== random-sphere figures ===\n');
    auspice_plot_journal(Ss{:}, ...
        'labels',       labels, ...
        'save_dir',     rs_save, ...
        'figure_list',  'full', ...
        'save_figures', true);
else
    fprintf(['\n[skip] random-sphere cells incomplete — run ' ...
             'reproduce_random_sphere_experiments first.\n']);
end

%% ---------------- Formation figures (4 cells) ----------------
fm_dir  = fullfile(root, 'paper_formation');
fm_save = fullfile(root, 'paper_figures', 'formation');

fm_cells = { ...
    struct('file', 'formation_pursuit_adv0p9__hocbf.mat',    'label', 'HOCBF\newlineslow pur'); ...
    struct('file', 'formation_pursuit_adv1p5__hocbf.mat',    'label', 'HOCBF\newlinefast pur'); ...
    struct('file', ['formation_pursuit_adv0p9__' nn_sfx],    'label', 'aTTC-CBF\newlineslow pur'); ...
    struct('file', ['formation_pursuit_adv1p5__' nn_sfx],    'label', 'aTTC-CBF\newlinefast pur')};

[Ss, labels, ok] = load_cells(fm_dir, fm_cells);
if ok
    fprintf('\n=== formation figures ===\n');
    auspice_plot_journal_formation(Ss{:}, ...
        'labels',       labels, ...
        'save_dir',     fm_save, ...
        'figure_list',  'full', ...
        'save_figures', true, ...
        'meb_xcap',     40, ...
        'meb_nbins',    200);
else
    fprintf(['\n[skip] formation cells incomplete — run ' ...
             'reproduce_formation_experiments first.\n']);
end
end


function [Ss, labels, ok] = load_cells(data_dir, cells)
%LOAD_CELLS  Load result cells, stripping each res to the fields the
%   plotters use (x, params, dist, u) to bound peak memory.
    n      = numel(cells);
    Ss     = cell(n, 1);
    labels = cell(n, 1);
    ok     = true;
    for k = 1:n
        path_k = fullfile(data_dir, cells{k}.file);
        if ~exist(path_k, 'file')
            fprintf('  missing: %s\n', path_k);
            ok = false;
            continue;
        end
        fprintf('Loading [%d/%d] %s\n', k, n, path_k);
        S = load(path_k, 'res', 'metrics');
        res_lean = struct('x', S.res.x, 'params', S.res.params);
        if isfield(S.res, 'dist'), res_lean.dist = S.res.dist; end
        if isfield(S.res, 'u'),    res_lean.u    = S.res.u;    end
        wrapper = struct('res', res_lean);
        if isfield(S, 'metrics'), wrapper.metrics = S.metrics; end
        Ss{k}     = wrapper;
        labels{k} = cells{k}.label;
        clear S res_lean
    end
end
