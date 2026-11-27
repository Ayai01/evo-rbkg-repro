#pragma once
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <ostream>
#include <vector>

namespace arh {
// Six electrical pin classes: heldQ=0/1, then CE=1 (oldQ,D)=00/01/10/11.
// These are captured-state responses. A source/receiver qualification owns
// charge, timing and physical support; the RTL does not invent probabilities.
struct CaptureClass { unsigned style=0,flips=0;bool current=false;std::array<int,2> pre{};std::array<int,6> absolute{}; };
class JointCapture {
    std::vector<CaptureClass> rows;
    std::vector<unsigned> word_styles;
    struct Evidence { unsigned cycle,domain,lane,bit,row,q,d,ce,expected,captured,pre;bool current; };
    std::vector<Evidence> evidence;
    std::array<std::uint64_t,6> pin_counts{};
    static std::string pending_value(const char*field,int status,unsigned index,unsigned q,unsigned cycle,unsigned domain,unsigned lane,unsigned bit) {
        return std::string("{")+"\"field\":\""+field+
            "\",\"status\":"+std::to_string(status)+",\"class\":"+std::to_string(index)+
            ",\"oldQ\":"+std::to_string(q)+",\"cycle\":"+std::to_string(cycle)+
            ",\"domain\":"+std::to_string(domain)+",\"lane\":"+std::to_string(lane)+",\"bit\":"+std::to_string(bit)+"}";
    }
    [[noreturn]] void pending(const char*field,int status,unsigned index,unsigned q,unsigned cycle,unsigned domain,unsigned lane,unsigned bit)const {
        throw std::runtime_error("Unresolved joint electrical response "+pending_value(field,status,index,q,cycle,domain,lane,bit));
    }
public:
    std::uint64_t checked=0,changed=0,pre_checked=0,pre_changed=0;
    void load(const std::vector<std::string>& f) {
        if(f.size()<2) throw std::invalid_argument("K requires response classes");
        auto n=std::stoul(f[1]);
        if(n<1 || n>4096 || f.size()!=2+2*n) throw std::invalid_argument("Invalid finite response table");
        rows.clear();
        for(unsigned i=0;i<n;++i) {
            unsigned style=std::stoul(f[2+2*i]),flips=std::stoul(f[3+2*i]);
            if(style>1 || flips>64) throw std::invalid_argument("Invalid physical response class");
            rows.push_back({style,flips});
        }
    }
    void load_current(const std::vector<std::string>&f) {
        if(f.size()<2)throw std::invalid_argument("C requires absolute current responses");
        auto n=std::stoul(f[1]);if(n<1||n>4096||f.size()!=2+9*n)throw std::invalid_argument("Invalid absolute current table");
        rows.clear();
        for(unsigned i=0;i<n;++i){CaptureClass row;row.current=true;row.style=std::stoul(f[2+9*i]);
            if(row.style>1)throw std::invalid_argument("Invalid physical current cell style");
            for(unsigned j=0;j<8;++j){int v=std::stoi(f[3+9*i+j]);if(v< -3||v>1)throw std::invalid_argument("Invalid absolute current bit");
                if(j<2)row.pre[j]=v;else row.absolute[j-2]=v;}
            rows.push_back(row);
        }
    }
    void reset() { checked=changed=pre_checked=pre_changed=0; evidence.clear();pin_counts.fill(0); }
    unsigned size()const { return rows.size(); }
    unsigned style(unsigned index)const { return rows.at(index).style; }
    bool current()const{return !rows.empty()&&rows.front().current;}
    void styles(const std::vector<std::string>& f,unsigned words) {
        if(f.size()!=2 || f[1].size()!=words)throw std::invalid_argument("One physical FF style per actual word required");
        word_styles.clear();
        for(char c:f[1]) {
            if(c!='0' && c!='1')throw std::invalid_argument("FF style must be standard or DICE");
            word_styles.push_back(c-'0');
        }
    }
    void preflight(std::vector<std::string>&missing,unsigned index,unsigned q,unsigned domain,unsigned cycle,unsigned lane,unsigned bit)const {
        const auto&row=rows.at(index);if(!row.current)return;
        if(row.style!=(word_styles.empty()?0:word_styles.at(domain)))throw std::runtime_error("Current class differs from actual selected FF style");
        if(row.pre.at(q)<0)missing.push_back(pending_value("pre_edge_bits",row.pre.at(q),index,q,cycle,domain,lane,bit));
    }
    static void require_preflight(const std::vector<std::string>&missing) {
        if(missing.empty())return;
        std::string value="Unresolved joint electrical response {\"contexts\":[";
        for(unsigned i=0;i<missing.size();++i){if(i)value+=',';value+=missing[i];}
        throw std::runtime_error(value+"]}");
    }
    unsigned pre(unsigned index,unsigned q,unsigned domain,unsigned cycle,unsigned lane,unsigned bit) {
        const auto&row=rows.at(index);if(!row.current)return q;
        if(row.style!=(word_styles.empty()?0:word_styles.at(domain)))throw std::runtime_error("Current class differs from actual selected FF style");
        if(row.pre.at(q)<0)pending("pre_edge_bits",row.pre.at(q),index,q,cycle,domain,lane,bit);
        ++pre_checked;unsigned value=unsigned(row.pre[q]);pre_changed+=value!=q;return value;
    }
    unsigned capture(unsigned index,unsigned q,unsigned d,bool ce,unsigned cycle,unsigned domain,unsigned lane,unsigned bit,int pre_value=-1) {
        const auto& row=rows.at(index);unsigned column=ce ? 2+2*q+d : q;
        if(row.style!=(word_styles.empty() ? 0 : word_styles.at(domain)))throw std::runtime_error("Response class differs from actual selected FF style");
        // Unsupported future events are harmless when the task has already
        // finished. Reached support gaps return an unresolved protocol error.
        if(row.current?row.absolute[column]<0:row.flips==64)pending("captured_bits",row.current?row.absolute[column]:-1,index,q,cycle,domain,lane,bit);
        unsigned before=pre_value<0?q:unsigned(pre_value),expected=ce?d:before;
        unsigned captured=row.current?unsigned(row.absolute[column]):expected^((row.flips>>column)&1U),flip=captured!=expected;
        ++checked;changed+=flip;++pin_counts[column];
        if(evidence.size()<8)evidence.push_back({cycle,domain,lane,bit,index,q,d,unsigned(ce),expected,captured,before,row.current});
        return captured;
    }
    void print(std::ostream& out)const {
        out<<",\"joint_capture_checks\":"<<checked<<",\"joint_response_changes\":"<<changed<<",\"joint_evidence\":[";
        for(unsigned i=0;i<evidence.size();++i) {
            if(i)out<<',';const auto&e=evidence[i];
            out<<"{\"cycle\":"<<e.cycle<<",\"domain\":"<<e.domain<<",\"lane\":"<<e.lane
               <<",\"bit\":"<<e.bit<<",\"class\":"<<e.row<<",\"oldQ\":"<<e.q<<",\"D\":"<<e.d
               <<",\"CE\":"<<e.ce<<",\"expected\":"<<e.expected<<",\"captured\":"<<e.captured;
            if(e.current)out<<",\"preQ\":"<<e.pre;out<<'}';
        }
        out<<"],\"joint_pin_class_counts\":[";
        for(unsigned i=0;i<6;++i) {if(i)out<<',';out<<pin_counts[i];}
        out<<']';
        if(current())out<<",\"joint_current_interface\":true,\"joint_pre_checks\":"<<pre_checked<<",\"joint_pre_changes\":"<<pre_changed;
    }
};
}
