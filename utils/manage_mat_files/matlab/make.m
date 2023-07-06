% change all .dat files to .mat files

x = getAllFiles('Sorted_alzh_dataset');
for c=1:length(x)
    if contains(x{c}, '.dat')
        file_name = x{c};
        data=load_bcidat(file_name);
        save(replace(file_name,'.dat','.mat'), 'data');
    end
end
