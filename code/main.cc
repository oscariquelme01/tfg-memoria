#include <iostream>
#include <ins_stitcher.h>

#include <iostream>
#include <algorithm>
#include <condition_variable>
#include <mutex>
#include <chrono>
#include <ostream>
#include <set>
#include <vector>
#include <sstream>
#include <filesystem>

#ifdef WIN32
#include <direct.h>
#include <Windows.h>
#endif // WIN32

using namespace std::chrono;
using namespace ins;

const std::string helpstr =
"{-help                   | default               | print this message                  }\n"
"{-inputs                 | None                  | input files                         }\n"
"{-stitch_type            | template              | template                            }\n"
"{                                                | optflow                             }\n"
"{                                                | dynamicstitch                       }\n"
"{                                                | aistitch                            }\n"
"{-ai_stitching_model     |                       | ai stitching model path             }\n"
"{-bitrate                | same as input video   | the bitrate of ouput file           }\n"
"{-enable_flowstate       | OFF                   | enable flowstate                    }\n"
"{-enable_directionlock   | OFF                   | enable directionlock                }\n"
"{-output_size            | 1920x960              | the resolution of output            }\n"
"{-enable_h265_encoder    | h264                  | encode format                       }\n"
"{-disable_cuda           | true                  | disable cuda                        }\n"
"{-enable_soft_encode     | false                 | use soft encoder                    }\n"
"{-enable_soft_decode     | false                 | use soft decoder                    }\n"
"{-enable_stitchfusion    | OFF                   | stitch_fusion                       }\n"
"{-enable_denoise         | OFF                   | enable denoise                      }\n"
"{-image_denoise_model    | OFF                   | image denoise model path            }\n"
"{-enable_colorplus       | OFF                   | enable colorplus                    }\n"
"{-colorplus_model        |                       | colorplus model path                }\n"
"{-enable_deflicker       | OFF                   | enable deflicker                    }\n"
"{-deflicker_model        |                       | deflicker model path                }\n"
"{-image_sequence_dir     | None                  | the output dir of image sequence    }\n"
"{-image_type             | jpg                   | jpg                                 }\n"
"{                                                | png                                 }\n"
"{-camera_accessory_type  | default 0             | refer to 'common.h'                 }\n"
"{-export_frame_index     |                       | Derived frame number sequence, example: 20-50-30 }\n";

const std::string RAW_SOURCES_BASE_PATH = "./sources/rawFootage";

// "_00_" or "_10_" are the insta 360 indicator patterns
std::string PATTERN_00 = "_00_";
std::string PATTERN_10 = "_10_";

std::string find_insta360_pair(std::string candidate) {

    size_t pos = candidate.find(PATTERN_00);
    if (pos == std::string::npos) {
        pos = candidate.find(PATTERN_10);

        if (pos == std::string::npos) return ""; // Neither pattern found -> no pair to be found

        // if the 10 pattern is found, replace it with pattern 00
        candidate.replace(pos, 4, PATTERN_00);
    } 
    else {
        // if the 00 pattern is found, replace it with pattern 10
        candidate.replace(pos, 4, PATTERN_10);
    }

    return candidate;
}

// Returns the insta 360 filename without the indicator patterns
std::string get_output_path(const std::string& filename) {
    std::filesystem::path p(filename);
    
    // Replace the parent directory and extension
    std::filesystem::path newPath = p.parent_path().parent_path() / "convertedFootage" / p.stem();
    newPath += ".mp4";
    
    return newPath.string();
}

bool are_insta360_pairs(const std::string& file1, const std::string& file2) {
    // Sanity check: lengths must be equal
    if (file1.length() != file2.length()) return false;

    size_t pos = file1.find(PATTERN_00);
    if (pos == std::string::npos) {
        pos = file1.find(PATTERN_10);
        if (pos == std::string::npos) return false; // Neither pattern found
    }

    // Check that at the same position, file2 has the *other* lens pattern
    // 4 is the length of the indicator pattern
    std::string lens1 = file1.substr(pos, 4);
    std::string lens2 = file2.substr(pos, 4);

    if ((lens1 == "_00_" && lens2 == "_10_") || (lens1 == "_10_" && lens2 == "_00_")) {
        // Temporarily replace lens pattern in both with same value to compare rest
        std::string normalized1 = file1;
        std::string normalized2 = file2;

        normalized1.replace(pos, 4, "_XX_");
        normalized2.replace(pos, 4, "_XX_");

        return normalized1 == normalized2;
    }

    return false;
}

static std::string stringToUtf8(const std::string& original_str) {
#ifdef WIN32
    const char* std_origin_str = original_str.c_str();
    const int std_origin_str_len = static_cast<int>(original_str.length());

    int nwLen = MultiByteToWideChar(CP_ACP, 0, std_origin_str, -1, NULL, 0);

    wchar_t* pwBuf = new wchar_t[nwLen + 1];
    ZeroMemory(pwBuf, nwLen * 2 + 2);

    MultiByteToWideChar(CP_ACP, 0, std_origin_str, std_origin_str_len, pwBuf, nwLen);

    int nLen = WideCharToMultiByte(CP_UTF8, 0, pwBuf, -1, NULL, NULL, NULL, NULL);

    char* pBuf = new char[nLen + 1];
    ZeroMemory(pBuf, nLen + 1);

    WideCharToMultiByte(CP_UTF8, 0, pwBuf, nwLen, pBuf, nLen, NULL, NULL);

    std::string ret_str(pBuf);

    delete[] pwBuf;
    delete[] pBuf;

    pwBuf = NULL;
    pBuf = NULL;

    return ret_str;
#else
    return original_str;
#endif
}

std::vector<std::string> split(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);

    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }

    return tokens;
}

std::string pick_inputs() {
    std::vector<std::string> files;
    
    try {
        // Check if directory exists
        if (!std::filesystem::exists(RAW_SOURCES_BASE_PATH) || !std::filesystem::is_directory(RAW_SOURCES_BASE_PATH)) {
            std::cout << "Directory does not exist: " << RAW_SOURCES_BASE_PATH << std::endl;
            return "";
        }

        std::set<std::string> seen_recordings;
        
        // Read all files from directory and filter the relevant ones
        for (const auto& entry : std::filesystem::directory_iterator(RAW_SOURCES_BASE_PATH)) {
            if (entry.is_regular_file()) {
                std::string filename = entry.path().filename().string();
                
                // Check if file ends with ".insv" and starts with "VID"
                if (filename.length() >= 5 && filename.substr(filename.length() - 5) == ".insv" &&
                    filename.length() >= 3 && filename.substr(0, 3) == "VID") {
                    
                    // The video names are formatted like "VID_{date}_{unique_id}_{lense_pattern_id}_{num_video}.insv"
                    // In order to not include two videos with the same date, unique_id and num_video but different lense_pattern_id
                    // Grab the unique id and push it to the already seen recording to keep track of which recording have already been pushed
                    std::vector<std::string> filename_tokens = split(filename, '_');
                    const std::string filename_id = filename_tokens[2];

                    // The videos need to be sorted (first the _00_, then t he _10_) which is why we will only show the _00_ pattern to pick
                    const int pos = filename.find(PATTERN_10);
                    if (pos != std::string::npos) {
                        filename.replace(pos, 4, PATTERN_00);
                    }

                    if (seen_recordings.find(filename_id) == seen_recordings.end()) {
                        seen_recordings.insert(filename_id);
                        files.push_back(filename);
                    }
                }
            }
        }
        
        // Check if any files were found
        if (files.empty()) {
            std::cout << "No files found in directory." << std::endl;
            return "";
        }
        
        // Display numbered list
        std::cout << "Files in directory:" << std::endl;
        for (size_t i = 0; i < files.size(); ++i) {
            std::cout << i + 1 << ". " << files[i] << std::endl;
        }
        
        // Get user selection
        int selection;
        std::cout << "Enter the number of the file you want to select (1-" << files.size() << "): ";
        std::cin >> selection;
        
        // Validate selection
        if (selection < 1 || selection > static_cast<int>(files.size())) {
            std::cout << "Invalid selection." << std::endl;
            return "";
        }
        
        // Return full path of selected file
        std::filesystem::path fullPath = std::filesystem::path(RAW_SOURCES_BASE_PATH) / files[selection - 1];
        return fullPath.string();
        
    } catch (const std::filesystem::filesystem_error& ex) {
        std::cout << "Filesystem error: " << ex.what() << std::endl;
        return "";
    }
}

int main(int argc, char* argv[]) {
    ins::SetLogLevel(ins::InsLogLevel::ERR);
    ins::InitEnv();

    std::vector<std::string> input_paths;
    std::string image_sequence_dir;
    std::string ai_stitching_model;
    std::string color_plus_model_path;
    std::string denoise_model_path;
    std::string exported_frame_number_sequence;
    std::string deflicker_model_path;

    STITCH_TYPE stitch_type = STITCH_TYPE::OPTFLOW;
    IMAGE_TYPE image_type = IMAGE_TYPE::JPEG;
    CameraAccessoryType accessory_type = CameraAccessoryType::kNormal;

    int output_width = 1920;
    int output_height = 960;
    int output_bitrate = 0;

    bool enable_flowstate = false;
    bool enable_cuda = true;
    bool enable_soft_encode = false;
    bool enable_soft_decode = false;
    bool enalbe_stitchfusion = true;
    bool enable_colorplus = false;
    bool enable_directionlock = false;
    bool enable_sequence_denoise = false;
    bool enable_H265_encoder = false;
    bool enable_deflicker = false;

    for (int i = 1; i < argc; i++) {
        if (std::string("-inputs") == std::string(argv[i])) {
            std::string input_path = argv[++i];

            // While no new argument is found
            while (input_path[0] != '-') {
                input_paths.push_back(stringToUtf8(input_path));
                input_path = argv[++i];
            }
        }

        if (std::string("-colorplus_model") == std::string(argv[i])) {
            color_plus_model_path = stringToUtf8(argv[++i]);
        }
        else if (std::string("-stitch_type") == std::string(argv[i])) {
            std::string stitchType = argv[++i];
            if (stitchType == std::string("optflow")) {
                stitch_type = STITCH_TYPE::OPTFLOW;
            }
            else if (stitchType == std::string("dynamicstitch")) {
                stitch_type = STITCH_TYPE::DYNAMICSTITCH;
            }
            else if (stitchType == std::string("aistitch")) {
                stitch_type = STITCH_TYPE::AIFLOW;
            }
        }
        else if (std::string("-enable_flowstate") == std::string(argv[i])) {
            enable_flowstate = true;
        }
        else if (std::string("-disable_cuda") == std::string(argv[i])) {
            enable_cuda = false;
        }
        else if (std::string("-enable_stitchfusion") == std::string(argv[i])) {
            enalbe_stitchfusion = true;
        }
        else if (std::string("-enable_denoise") == std::string(argv[i])) {
            enable_sequence_denoise = true;
        }
        else if (std::string("-enable_colorplus") == std::string(argv[i])) {
            enable_colorplus = true;
        }
        else if (std::string("-enable_directionlock") == std::string(argv[i])) {
            enable_directionlock = true;
        }
        else if (std::string("-enable_h265_encoder") == std::string(argv[i])) {
            enable_H265_encoder = true;
        }
        else if (std::string("-bitrate") == std::string(argv[i])) {
            output_bitrate = atoi(argv[++i]);
        }
        else if (std::string("-output_size") == std::string(argv[i])) {
            auto res = split(std::string(argv[++i]), 'x');
            if (res.size() == 2) {
                output_width = std::atoi(res[0].c_str());
                output_height = std::atoi(res[1].c_str());
            }
        }
        else if (std::string("-image_sequence_dir") == std::string(argv[i])) {
            image_sequence_dir = std::string(argv[++i]);
        }
        else if (std::string("-image_type") == std::string(argv[i])) {
            std::string type = argv[++i];
            if (type == std::string("jpg")) {
                image_type = IMAGE_TYPE::JPEG;
            }
            else if (type == std::string("png")) {
                image_type = IMAGE_TYPE::PNG;
            }
        }
        else if (std::string("-camera_accessory_type") == std::string(argv[i])) {
            accessory_type = static_cast<CameraAccessoryType>(std::atoi(argv[++i]));
        }
        else if (std::string("-ai_stitching_model") == std::string(argv[i])) {
            ai_stitching_model = stringToUtf8(argv[++i]);
        }
        else if (std::string("-image_denoise_model") == std::string(argv[i])) {
            denoise_model_path = stringToUtf8(argv[++i]);
        }
        else if (std::string("-export_frame_index") == std::string(argv[i])) {
            exported_frame_number_sequence = argv[++i];
        }
        else if (std::string("-deflicker_model") == std::string(argv[i])) {
            deflicker_model_path = stringToUtf8(argv[++i]);
        }
        else if (std::string("-enable_deflicker") == std::string(argv[i])) {
            enable_deflicker = true;
        }
        else if (std::string("-enable_soft_encode") == std::string(argv[i])) {
            enable_soft_encode = true;
        }
        else if (std::string("-enable_soft_decode") == std::string(argv[i])) {
            enable_soft_decode = true;
        }
        else if (std::string("-help") == std::string(argv[i])) {
            std::cout << helpstr << std::endl;
        }
    }

    if (!input_paths.size()) {
        input_paths.resize(1);
        input_paths[0] = pick_inputs();

        if (input_paths[0].empty()) {
            std::cout << "Failed to get the chosen input";
            return -1;
        }
    }

    // either the user only specified 1 input file or no inputs file and used the interactive selector
    // either way we will try to find the insta360 video pair
    if (input_paths.size() == 1) {
        input_paths.resize(2);
        input_paths[1] = find_insta360_pair(input_paths[0]);
        if (input_paths[1].empty()) {
            std::cout << "Failed to get the pair video for the specified input";
            return -1;
        }
    }

    // check for incorrect input parameters
    if (input_paths.size() != 2) {
        std::cout << "You need to specify a single pair of .insv videos. Multiple files stitching not yet supported" << std::endl;
        std::cout << helpstr << std::endl;
        return -1;
    }

    std::cout << "Inputs:" << std::endl;
    std::cout << input_paths[0] << std::endl;
    std::cout << input_paths[1] << std::endl << std::flush;

    if (!are_insta360_pairs(input_paths[0], input_paths[1])) {
        std::cout << "The specified pair doesn't seem to be from the same recording. Ensure the names are correct with the _00_ or _10_ indicator patterns" << std::endl;
        std::cout << helpstr << std::endl;
        return -1;
    }

    const std::string output_path = get_output_path(input_paths[0]);
    if (output_path.empty()) {
        // this should never happend actually because we have previously checked
        std::cout << "Failed to get the output filename" << std::endl;
        std::cout << helpstr << std::endl;
        return -1;
    }

    std::cout << "Output: \n";
    std::cout << output_path << std::endl;

    std::vector<uint64_t> export_frame_nums;
    if (!image_sequence_dir.empty()) {
        auto frame_index_vec = split(exported_frame_number_sequence, '-');
        for (auto& frame_index : frame_index_vec) {
            int index = atoi(frame_index.c_str());
            export_frame_nums.push_back(index);
        }
    }

    if (color_plus_model_path.empty()) {
        enable_colorplus = false;
    }

    int count = 1;
    while (count--) {
        std::mutex mutex;
        std::condition_variable cond;
        bool is_finished = false;
        bool has_error = false;
        int stitch_progress = 0;
        std::string suffix = input_paths[0].substr(input_paths[0].find_last_of(".") + 1);
        std::transform(suffix.begin(), suffix.end(), suffix.begin(), ::tolower);

        auto start_time = steady_clock::now();
        auto video_stitcher = std::make_shared<VideoStitcher>();
        video_stitcher->SetInputPath(input_paths);
        if (image_sequence_dir.empty()) {
            video_stitcher->SetOutputPath(output_path);
        }
        else {
            if (!export_frame_nums.empty()) {
                video_stitcher->SetExportFrameSequence(export_frame_nums);
            }

            video_stitcher->SetImageSequenceInfo(image_sequence_dir, image_type);
        }
        video_stitcher->SetStitchType(stitch_type);
        video_stitcher->EnableCuda(enable_cuda);
        video_stitcher->EnableStitchFusion(enalbe_stitchfusion);
        video_stitcher->EnableColorPlus(enable_colorplus, color_plus_model_path);
        video_stitcher->SetOutputSize(output_width, output_height);
        video_stitcher->SetOutputBitRate(output_bitrate);
        video_stitcher->EnableFlowState(enable_flowstate);
        video_stitcher->SetAiStitchModelFile(ai_stitching_model);
        video_stitcher->EnableDenoise(enable_sequence_denoise);
        video_stitcher->EnableDirectionLock(enable_directionlock);
        video_stitcher->SetCameraAccessoryType(accessory_type);
        video_stitcher->SetSoftwareCodecUsage(enable_soft_encode, enable_soft_decode);
        if (enable_H265_encoder) {
            video_stitcher->EnableH265Encoder();
        }
        video_stitcher->EnableDeflicker(enable_deflicker, deflicker_model_path);
        video_stitcher->SetStitchProgressCallback([&](int process, int error) {
            if (stitch_progress != process) {
                const std::string process_desc = "process = " + std::to_string(process) + std::string("%");
                std::cout << "\r" << process_desc << std::flush;
                stitch_progress = process;
            }

            if (stitch_progress == 100) {
                std::cout << std::endl;
                std::unique_lock<std::mutex> lck(mutex);
                cond.notify_one();
                is_finished = true;
            }
        });

        video_stitcher->SetStitchStateCallback([&](int error, const char* err_info) {
            std::cout << "error: " << err_info << std::endl;
            has_error = true;
            cond.notify_one();
        });

        std::cout << "start stitch " << std::endl;
        video_stitcher->StartStitch();

        std::unique_lock<std::mutex> lck(mutex);
        cond.wait(lck, [&] {
            std::cout << "progress: " << video_stitcher->GetStitchProgress() << "; finished: " << is_finished << std::endl;
            return is_finished || has_error;
        });

        std::cout << "end stitch " << std::endl;

        auto end_time = steady_clock::now();
        std::cout << "cost = " << duration_cast<duration<double>>(end_time - start_time).count() << std::endl;
    }
    return 0;
}
