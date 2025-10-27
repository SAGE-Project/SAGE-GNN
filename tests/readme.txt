# Test Generator (`test.py`)
test.py script generates **15 test cases** that simulate component-to-VM allocation scenarios with randomized hardware
requirements and VM specifications.

## 📌 Test Generation
* **Number of Components:** Randomly chosen between **15 and 20**.
* **Number of VMs:** Randomly chosen between **15 and 20**.
* **Constraints:**
* **No inter-component constraints** (components are independent).
* Only **hardware requirements** and **VM provisioning** constraints are considered.

## ⚙️ Parameters
* **Component Hardware Requirements:**
  * CPU
  * Memory
  * Storage

* **VM Specifications:**
  * CPU
  * Memory
  * Storage
  * Price

All these values are **randomly generated integers** for each test case.

## 📂 Directory Structure
* **`inputs/`** – Contains generated test input files for each test case.
* **`outputs/`** – Contains the generated output files with the initial allocation results.

### 🔄 `test-set-init-valsX.py`

* File in `outputs/` contains the **initial solution or baseline allocation** produced by `test.py`, except those for which a result
was not found in > 5800 seconds
* These results were later **used as predefined initial values** when creating the scripts:
  ```
  test-set-init-vals1.py
  test-set-init-vals2.py
  test-set-init-vals6.py
  test-set-init-vals8.py
  test-set-init-vals12.py
  test-set-init-vals15.py
* Essentially, the outputs generated here **served as the starting point** for further experiments or simulations
 in these `test-set-init-valsX.py` files.