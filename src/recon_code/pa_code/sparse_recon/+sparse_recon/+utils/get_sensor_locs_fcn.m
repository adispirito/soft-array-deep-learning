function [x0, y0, z0] = get_sensor_locs_fcn(transParaPath)
    x0 = load([transParaPath,'\','x_trans.mat']); x0 = x0.x_trans;
    y0 = load([transParaPath,'\','y_trans.mat']); y0 = y0.y_trans;
    z0 = load([transParaPath,'\','z_trans.mat']); z0 = z0.z_trans;
end