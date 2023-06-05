package capstone.book_grim_ai;

import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;

import java.io.*;
import java.util.ArrayList;

class BookGrimAiApplicationTests {

	@Test
	void readSpell() throws IOException {
		File file = new File("E:\\2023.1\\캡스톤\\py-hanspell\\spellList.txt");
		if(file.exists())
		{
			ArrayList<String> spellList = new ArrayList<String>();
			BufferedReader inFile = new BufferedReader(new InputStreamReader(new FileInputStream(file), "utf-8"));
			String sLine = "";

			while( (sLine = inFile.readLine()) != null ){
				spellList.add(sLine);
			}
			if (spellList.isEmpty()){
				System.out.println("Empty!");
			}
			else{
				for(String o : spellList){
					System.out.println(o);
				}
			}
		}
	}
}
