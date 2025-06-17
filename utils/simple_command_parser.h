//
// Created by sunsx on 04/06/21.
// Modified by gli945 on 16/01/25.
//

#ifndef RAPIDMATCH_SIMPLE_COMMAND_PARSER_H
#define RAPIDMATCH_SIMPLE_COMMAND_PARSER_H

#include <algorithm>
#include <numeric>
#include <string>
#include <vector>

class InputParser {
  public: 
    InputParser (int &argc, char **argv){
        for (int i = 1; i < argc; ++i)
            tokens_.emplace_back(argv[i]);
    }

    std::string get_cmd_option(const std::string &option) const{
        std::vector<std::string>::const_iterator itr;
        itr =  std::find(tokens_.begin(), tokens_.end(), option);
        if (itr != tokens_.end() && ++itr != tokens_.end()){
            return *itr;
        }
        return "";
    }

    // Added by gli945
    int32_t get_int32_cmd_option(const std::string &option, int32_t default_value=0) const {
        std::string string_value = get_cmd_option(option);
        if (string_value.empty()) {
            return default_value;
        }
        return std::stoi(string_value);
    }

    // Added by gli945
    uint32_t get_uint32_cmd_option(const std::string &option, uint32_t default_value=UINT32_MAX) const {
        std::string string_value = get_cmd_option(option);
        if (string_value.empty()) {
            return default_value;
        }
        return std::stoul(string_value);
    }

    // Added by gli945
    bool get_bool_cmd_option(const std::string &option, bool default_value=false) const {
        std::string string_value = get_cmd_option(option);
        if (string_value.empty()) {
            return default_value;
        } else if (string_value == "true" || string_value == "True") {
            return true;
        } else if (string_value == "false" || string_value == "False") {
            return false;
        } else {
            return default_value;
        }
    }

    bool check_cmd_option_exists(const std::string &option) const{
        return std::find(tokens_.begin(), tokens_.end(), option)
               != tokens_.end();
    }

    // Modified by gli945
    std::string get_cmd() {
        return std::accumulate(tokens_.begin(), tokens_.end(), std::string("csmg"),
                                         [](std::string &a, std::string &b) -> std::string {
                                             return a + " " + b;
                                         });
    }

  private:
    std::vector<std::string> tokens_;
};

#endif //RAPIDMATCH_SIMPLE_COMMAND_PARSER_H
