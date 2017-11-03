-- dataloader
-- todo
-- sampler
-- labeler

require 'torch'
require 'image'
require 'sys'
require 'xlua'
local ffi = require 'ffi'
local class = require('pl.class')
local dir = require('pl.dir')
local argcheck = require 'argcheck'
local prepro = require 'data.preprocessor'



-- input arguments check. if not exist, set as default value.
function argchecker(param)

    -- set default value.
    param.loadSize = param.loadSize or 80
    param.sampleSize = param.sampleSize or 64
    param.batchSize = param.batchSize or 10
    param.nthreads = param.nthreads or 8
    param.trainPath = param.trainPath or nil
    param.split = param.split or 1.0
    param.channel = param.channel or 3
    param.verbose = param.verbose or true
    param.hflip = param.hflip or false
    param.crop = param.crop or 'random'
    param.padding = param.padding or false
    param.keep_ratio = param.keep_ratio or true
    param.pixel_range = param.pixel_range or '[0,1]'
    param.rotate = param.rotate or 0
    param.whitenoise = param.whitenoise or 0
    param.brightness = param.brightness or -1

    -- arg type check
    assert(type(param)=='table')
    assert(type(param.loadSize)=='number')
    assert(type(param.sampleSize)=='number')
    assert(type(param.batchSize)=='number')
    assert(type(param.nthreads)=='number')
    assert(type(param.trainPath)=='string')
    assert(type(param.split)=='number')
    assert(type(param.channel)=='number')
    assert(param.channel==1 or param.channel==3)
    assert(type(param.verbose)=='boolean')
    assert(type(param.hflip)=='boolean')
    assert(type(param.padding)=='boolean')
    assert(type(param.keep_ratio)=='boolean')
    assert(type(param.crop)=='string')
    assert(param.crop=='center' or param.crop=='random' or param.crop=='none')
    assert(type(param.pixel_range)=='string')
    assert(param.pixel_range=='[0,1]' or param.pixel_range=='[-1,1]')
    assert(type(param.rotate)=='number')
    assert(type(param.whitenoise)=='number')
    assert(type(param.brightness)=='number')

    return param
end


-- basic settings
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)


-- dataloader
local dataloader = torch.class('DataLoader')


function dataloader:__init(...)

    -- argcheck
    local args = argchecker(...)
    for k,v in pairs(args) do self[k] = v end          -- push args into self.
    

    if not paths.dirp(self.trainPath) then
        error(string.format('Did not find directory: %s', self.trainPath))
    end

    -- renew cache to reload loader parameters.
    --os.execute('rm -rf cache')
    -- a cache file of the training metadata (if doesn't exist, will be created)
    local cache = "cache"
    local cache_prefix = self.trainPath:gsub('/', '_')
    os.execute('mkdir -p cache')
    local trainCache = paths.concat(cache, cache_prefix .. '_trainCache.t7')

    -- cache
    if paths.filep(trainCache) then 
        print('Loading train metadata from cache')
        info = torch.load(trainCache)
        for k,v in pairs(info) do 
            if self[k]==nil then self[k] = v end       -- restore variable.
        end
    else
        print('Creating train metadata')
        self:create_cache()
        torch.save(trainCache, self)
    end
end



function dataloader:size(target, class)
    assert( target=='train' or target=='test' or target=='all' or target==nil,
            'options : train | test | all')
    
    if target == 'train' then
        local list = self.classListTrain
        if not class then return self.numTrainSamples
        elseif type(class)=='string' then return list[self.classIndices[class]]:size(1)
        elseif type(class)=='number' then return list[class]:size(1) end
    elseif target == 'test' then
        local list = self.classListTest
        if not class then return self.numTestSamples
        elseif type(class)=='string' then return list[self.classIndices[class]]:size(1)
        elseif type(class)=='number' then return list[class]:size(1) end
    elseif target == 'all' or target==nil then
        local list = self.classList
        if not class then return self.numSamples
        elseif type(class)=='string' then return list[self.classIndices[class]]:size(1)
        elseif type(class)=='number' then return list[class]:size(1) end
    end
end

function dataloader:load_im(path)
    return image.load(path, nc, 'float')
end

-- all image processings for single image are defined here. (train)
function dataloader:trainHook(path)
    local src = self:load_im(path)
    local im = src:clone()
    
    -- add padding if want.
    if self.padding then im = prepro.add_padding(im, self.loadSize) end

    -- resize image (always keeping ratio)
    im = prepro.im_resize(im, self.loadSize, self.keep_ratio)

    -- crop image (random | center).
    im = prepro.im_crop(im, self.sampleSize, self.crop)

    -- rotation
    if self.rotate ~= 0 then
        im = prepro.im_rotate(im, tonumber(self.rotate))
    end
    
    -- hflip
    if self.hflip then im = prepro.im_hflip(im) end

    -- add whitenoise
    if self.whitenoise ~= 0 then im = prepro.im_add_noise(im, self.whitenoise) end

    -- add random brightness
    if self.brightness~=0 then im = prepro.im_brightness(im, self.brightness) end

    -- pixel range
    im = prepro.adjust_range(im, self.pixel_range)


    collectgarbage()
    return im
end


function dataloader:testHook(path)
    collectgarbage()
end


function dataloader:getByClass(target, class)
    assert(target=='train' or target=='test')
    local index, imgpath
    if target == 'train' then
        assert(self.numTrainSamples>0,
                    'no train samples found! recommend to check "split" option.')
        index = math.ceil(torch.uniform()*self.classListTrain[class]:nElement())
        imgpath = ffi.string(torch.data(self.imagePath[self.classListTrain[class][index]]))
        return self:trainHook(imgpath)
    elseif target == 'test' then
        assert(self.numTestSamples>0,
                    'no test samples found! recommend to check "split" option.')
        index = math.ceil(torch.uniform()*self.classListTest[class]:nElement())
        imgpath = ffi.string(torch.data(self.imagePath[self.classListTest[class][index]]))
        return self:testHook(imgpath)
    end
end

function dataloader:getSample(target)
    assert(target == 'train' or target == 'test')
    local c = torch.random(1, #self.classes)
    return self:getByClass(target, c)
end


function dataloader:sample(target, batchSize)
    print('gogogo')
    assert(batchSize, 'batchSize is not found.')
    local dataTable = {}
    local scalarTable = {}
    for i=1,batchSize do
        local class = torch.random(1,#self.classes)
        local out = self:getByClass(target, class)
        table.insert(dataTable, out)
        table.insert(scalarTable, class)
    end
    local data, scalarLabels = self:table2Output(self, dataTable, scalarTable)
    return data, scalarLabels
end

-- dataTable : images of batch, scalarTable: labels of batch.
function dataloader:table2Output(self, dataTable, scalarTable)
    local data, scalarLabels, labels
    local batchSize = #scalarTable
    assert(dataTable[1]:dim()==3, 'error. Size of input tensor must be 3.')
    data = torch.Tensor(batchSize, self.channel, self.sampleSize, self.sampleSize)
    scalarLabels = torch.LongTensor(batchSize):fill(-1)
    for i=1, #dataTable do
        data[i]:copy(dataTable[i])
        scalarLabels[i] = scalarTable[i]
    end
    return data, scalarLabels
end




function dataloader:create_cache()
    self.classes = {}
    local classPaths = {}
    
    -- search and return key value of table element.
    local function tableFind(t,o) for k,v in pairs(t) do if v == o then return k end end end
    
    -- loop over each paths folder, get list of unique class names.
    -- also stor the directory paths per class
    for k, path in ipairs({self.trainPath}) do
        local dirs = dir.getdirectories(path)
        for k, dirpath in ipairs(dirs) do
            local class = paths.basename(dirpath)            -- set class name (folder basename)
            local idx = tableFind(self.classes, class)      -- find class idx
            if not idx then
                table.insert(self.classes, class)         -- insert class names
                idx = #self.classes                         -- indexing
                classPaths[idx] = {}                   
            end
            if not tableFind(classPaths[idx], dirpath) then
                table.insert(classPaths[idx], dirpath)      -- save path info for each class.
            end
        end
    end

    self.classIndices = {}
    for k, v in ipairs(self.classes) do self.classIndices[v] = k end    -- indexing w.r.t class name
    
    -- define command-line tools.
    local wc = 'wc'
    local cut = 'cut'
    local find = 'find -H'   -- if folder name is symlink, do find inside it after dereferencing
    if jit.os == 'OSX' then
        wc = 'gwc'
        cut = 'gcut'
        find = 'gfind'
    end

    -- Options for the GNU find command
    local extList = {'jpg','jpeg','png','JPG','PNG','JPEG','ppm','PPM','bmp','BMP'}
    local findOptions = ' -iname "*.' .. extList[1] .. '"'
    for i=2,#extList do findOptions = findOptions .. ' -o -iname "*.' .. extList[i] .. '"' end
    
    -- find the image path names
    self.imagePath = torch.CharTensor()
    self.imageClass = torch.LongTensor()
    self.classList = {}
    self.classListSample = self.classList

    print(   'running "find" on each class directory, and concatenate all'
          .. ' those filenames into a single file containing all image paths for a given class')

    -- so, generate one file per class
    local classFindFiles = {}
    for i=1, #self.classes do classFindFiles[i] = os.tmpname() end
    local combinedFindList = os.tmpname()       -- will combine all here.
    
    local tmpfile = os.tmpname()
    local tmphandle = assert(io.open(tmpfile, 'w'))
    -- iterate over classes
    for i, class in ipairs(self.classes) do
        -- iterate over classPaths
        for j, path in ipairs(classPaths[i]) do
            local command = find .. ' "' .. path .. '" ' .. findOptions
                            .. ' >>"' .. classFindFiles[i] .. '" \n'
            tmphandle:write(command)
        end
    end
    io.close(tmphandle)
    os.execute('bash ' .. tmpfile)
    os.execute('rm -f ' .. tmpfile)
    
    collectgarbage()
    print('now combine all the files to a single large file')
    local tmpfile = os.tmpname()
    local tmphandle = assert(io.open(tmpfile, 'w'))
    -- cocat all finds to a single large file in the order of self.classes
    for i = 1, #self.classes do
        local command = 'cat "' .. classFindFiles[i] .. '" >>' .. combinedFindList .. ' \n'
        tmphandle:write(command)
    end
    io.close(tmphandle)
    os.execute('bash ' .. tmpfile)
    os.execute('rm -f ' .. tmpfile)


    -- now we have the large concatenated list of sampel paths. let's push it to self.imagPath!
    print('load the large concatenated list of sample paths to self.imagePath')
    local maxPathLength = tonumber(sys.fexecute(wc .. " -L '"
                                                   .. combinedFindList .. "' |"
                                                   .. cut .. " -f1 -d' '")) + 1
    local length = tonumber(sys.fexecute(wc .. " -l '"
                                            .. combinedFindList .. "' |"
                                            .. cut .. " -f1 -d' '"))
    assert(length > 0, "Could not find any image file in the give input paths")
    assert(maxPathLength > 0, "paths of files are length 0?")
    self.imagePath:resize(length, maxPathLength):fill(0)
    local s_data = self.imagePath:data()        -- return cdata char.
    local cnt = 0
    for line in io.lines(combinedFindList) do
        ffi.copy(s_data, line)                  -- ffi enables LuaJIT speed access to Tensors.
        s_data = s_data + maxPathLength
        if self.verbose and cnt%100 == 0 then
            xlua.progress(cnt,length)
        end
        cnt = cnt + 1
    end

    self.numSamples = self.imagePath:size(1)
    if self.verbose then print(self.numSamples .. ' samples found.') end
    
    -- now, we are going to update classList and imageClass.
    print('Updating classLIst and image Class appropriately')
    self.imageClass:resize(self.numSamples)
    local runningIndex = 0
    for i=1, #self.classes do
        if self.verbose then xlua.progress(i, #self.classes) end
        local length = tonumber(sys.fexecute(wc .. " -l '"
                                                .. classFindFiles[i] .. "' |"
                                                .. cut .. " -f1 -d' '"))
        if length == 0 then
            error('Class has zero samples')
        else
            self.classList[i] = torch.linspace(runningIndex + 1, runningIndex + length, length):long()
            self.imageClass[{{runningIndex+1, runningIndex+length}}]:fill(i)
        end
        runningIndex = runningIndex + length
    end

    -- clean up temp files.
    collectgarbage()
    print('Cleaning up temporary files')
    local tmpfilelistall = ''
    for i=1,#classFindFiles do
        tmpfilelistall = tmpfilelistall .. ' "' .. classFindFiles[i] .. '"'
        if i % 1000 == 0 then
            os.execute('rm -f ' .. tmpfilelistall)
            tmpfilelistall = ''
        end
    end
    os.execute('rm -f '  .. tmpfilelistall)
    os.execute('rm -f "' .. combinedFindList .. '"')

    -- split train/test set.
    local totalTestSamples = 0
    if self.split >= 1.0 then self.split=1.0 end
    
    print(string.format('Splitting train and test sets to a ratio of %.2f(train) / %.2f(test)',
            self.split, 1.0-self.split))
    self.classListTrain = {}
    self.classListTest = {}
    self.classListSample = self.classListTrain
    -- split the classList into classListTrain and classListTest
    for i=1,#self.classes do
        local list = self.classList[i]
        local count = self.classList[i]:size(1)
        local splitidx = math.floor(count*self.split + 0.5)     -- +round
        local perm = torch.randperm(count)                      -- mix (1 ~ count) randomly.
        self.classListTrain[i] = torch.LongTensor(splitidx)
        for j = 1, splitidx do                  -- (1 ~ splitidx) : trainset
            self.classListTrain[i][j] = list[perm[j]]
        end
        if splitidx == count then               -- all smaples were allocated to trainset
            self.classListTest[i] = torch.LongTensor()
        else 
            self.classListTest[i] = torch.LongTensor(count-splitidx)
            totalTestSamples = totalTestSamples + self.classListTest[i]:size(1)
            local idx = 1
            for j = splitidx+1, count do        -- (splitidx+1 ~ count) : testset
                self.classListTest[i][idx] = list[perm[j]]
                idx = idx + 1
            end
        end
    end
    -- Now combine classListTest into a single tensor
    collectgarbage()
    self.testIndices = torch.LongTensor(totalTestSamples)
    self.numTestSamples = totalTestSamples
    self.numTrainSamples = self.numSamples - self.numTestSamples
    local tdata = self.testIndices:data()
    local tidx = 0
    for i=1,#self.classes do
        local list = self.classListTest[i]
        if list:dim() ~= 0 then
            local ldata = list:data()
            for j=0, list:size(1)-1 do
                tdata[tidx] = ldata[j]
                tidx = tidx + 1
            end
        end
    end
end


return dataloader








