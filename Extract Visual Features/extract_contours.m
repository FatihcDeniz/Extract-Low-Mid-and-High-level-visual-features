% clear everything
close all
clear all
% image location and save location

addpath(genpath(".\utils\MLVcode"))

root = "..\Images";
save_loc_mid = "C:\Users\20225378\Desktop\Extract Visual Features\Extracted Visual Features\Mid";

% Get all items in root folder
folders = dir(root);

% Keep only folders (remove '.' '..' and files)
folders = folders([folders.isdir]);
folders = folders(~ismember({folders.name}, {'.', '..'}));

T_cell = cell(length(folders), 1);

for k = 1:length(folders)

    % current folder path
    image_folder = fullfile(folders(k).folder, folders(k).name);

    % get all files inside this folder
    img_files = dir(fullfile(image_folder, '*'));
    img_files = img_files(~[img_files.isdir]); % remove subfolders

    % keep only image files
    valid_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.gif'};
    is_img = false(length(img_files), 1);

    for n = 1:length(img_files)
        [~, ~, ext] = fileparts(img_files(n).name);
        is_img(n) = ismember(lower(ext), valid_ext);
    end

    img_files = img_files(is_img);

    T_local = table;

    for j = 1:length(img_files)

        image_loc = fullfile(img_files(j).folder, img_files(j).name);
        disp(image_loc)

        try
            % extract image name (without extension)
            [~, image_name, ~] = fileparts(img_files(j).name);
            T_local.image(j) = string(img_files(j).name);

            % read + process image
            img = imread(image_loc);

            % resize to 224x224
            img = imresize(img, [224 224]);

            cute = traceLineDrawingFromRGB(image_loc, img);

            curvature = getCurvatureStats(cute).normSumCurvatureHistogram;
            orientation = getOrientationStats(cute).normSumOrientationHistogram;
            clength = getLengthStats(cute).normSumLengthHistogram;

            for i = 1:8
                T_local{j, "curvature"+i} = curvature(i);
                T_local{j, "orientation"+i} = orientation(i);
                T_local{j, "length"+i} = clength(i);
            end

        catch ME
            warning('Skipping file %s: %s', image_loc, ME.message);
        end
    end

    T_cell{k} = T_local;
end

T = vertcat(T_cell{:});
writetable(T, fullfile(save_loc_mid, 'contour.csv'));