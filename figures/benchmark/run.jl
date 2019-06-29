function make_script()
	sample_size = collect(10000:10000:120000)
	for samples in sample_size
		open("hi.sh", "w") do file
    		write(file, "#!/bin/bash")
		end
	end
end