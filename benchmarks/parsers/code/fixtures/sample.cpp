#include <iostream>
#include <vector>

class VectorBuffer {
public:
    void push(int value) {
        items.push_back(value);
    }

    size_t size() const {
        return items.size();
    }
private:
    std::vector<int> items;
};